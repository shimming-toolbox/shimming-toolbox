#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import scipy.linalg
import logging
from typing import List

from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.coordinates import resample_from_to

ListCoil = List[Coil]
logger = logging.getLogger(__name__)


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
    Use optimize(args) to optimize a given mask.
    For basic optimizer, uses *unbounded* pseudo-inverse.

    Attributes:
        coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
        unshimmed (np.ndarray): 3d array of unshimmed volume
        unshimmed_affine (np.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
        merged_coils (np.ndarray): 4d array containing all coil profiles resampled onto the target unshimmed array
                                      concatenated on the 4th dimension. See self.merge_coils() for more details
        merged_bounds (list): list of bounds corresponding to each merged coils: merged_bounds[3] is the (min, max)
                              bound for merged_coils[..., 3]
    """

    def __init__(self, coils: ListCoil, unshimmed, affine):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        """
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        self.coils = coils
        self.unshimmed = np.array([])
        self.unshimmed_affine = []
        self.merged_coils = []
        self.merged_bounds = []
        self.set_unshimmed(unshimmed, affine)

    def set_unshimmed(self, unshimmed, affine):
        """
        Set the unshimmed array to a new array. Resamples coil profiles accordingly.

        Args:
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine: (np.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
        """
        # Check dimensions of unshimmed map
        if unshimmed.ndim != 3:
            raise ValueError(f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (dim1, dim2, dim3)")

        # Check dimensions of affine
        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")

        # Define coil profiles if unshimmed or affine is different than previously
        if (self.unshimmed.shape != unshimmed.shape) or not np.all(self.unshimmed_affine == affine):
            self.merged_coils, self.merged_bounds = self.merge_coils(unshimmed, affine)

        self.unshimmed = unshimmed
        self.unshimmed_affine = affine

    def set_merged_bounds(self, merged_bounds):
        """
        Changes the default bounds set in the coil profile

        Args:
            merged_bounds: Concatenated coil profile bounds
        """
        if len(self.merged_bounds) != len(merged_bounds):
            raise ValueError(f"Size of merged bounds: must match the number of total channel: {len(self.merged_bounds)}")
        self.merged_bounds = merged_bounds

    def optimize(self, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (np.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                                  unshimmed

        Returns:
            np.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """
        coil_mat, unshimmed_vec = self.get_coil_mat_and_unshimmed(mask)

        output = -1 * scipy.linalg.pinv(coil_mat) @ unshimmed_vec  # N x mV' @ mV'

        return output

    def get_coil_mat_and_unshimmed(self, mask):
        """
        Returns the coil matrix, and the unshimmed vector used for the optimization

        Args:
            mask (np.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                              unshimmed
        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 2D flattened array (point, channel) of masked coils
                              (axis 0 must align with unshimmed_vec)
                * np.ndarray: 1D flattened array (point) of the masked unshimmed map

        """
        # Check for sizing errors
        self._check_sizing(mask)
        # Define coil profiles
        n_channels = self.merged_coils.shape[3]
        mask_vec = mask.reshape((-1,))
        # Reshape coil profile: X, Y, Z, N --> N, X, Y, Z --> N, [mask.shape]
        # --> N, mask.size --> mask.size, N --> masked points, N
        merged_coils_reshaped = np.reshape(np.transpose(self.merged_coils, axes=(3, 0, 1, 2)),
                                           (n_channels, -1))
        masked_points_indices = np.where(mask_vec != 0)

        coil_mat = merged_coils_reshaped[:, masked_points_indices[0]].T  # masked points x N
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[masked_points_indices[0]]  # mV'

        return coil_mat, unshimmed_vec

    def merge_coils(self, unshimmed, affine):
        """
        Uses the list of coil profiles to return a resampled concatenated list of coil profiles matching the
        unshimmed image. Bounds are also concatenated and returned.

        Args:
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        """

        coil_profiles_list = []

        # Define the nibabel unshimmed array
        nii_unshimmed = nib.Nifti1Image(unshimmed, affine)

        # # Make sure all the coils have the same units
        # units = [coil.units for coil in self.coils]
        # if units.count(units[0]) != len(units):
        #     names = [coil.name for coil in self.coils]
        #     logger.warning(f"The coils don't have matching units: {list(zip(names, units))}")

        for coil in self.coils:
            nii_coil = nib.Nifti1Image(coil.profile, coil.affine)

            # Resample a coil on the unshimmed image
            resampled_coil = resample_from_to(nii_coil, nii_unshimmed).get_fdata()
            coil_profiles_list.append(resampled_coil)

        coil_profiles = np.concatenate(coil_profiles_list, axis=3)

        bounds = self.merge_bounds()

        return coil_profiles, bounds

    def merge_bounds(self):
        """
        Merge the coil profile bounds into a single array.

        Returns:
            list: list of bounds corresponding to each merged coils
        """

        bounds = []
        for coil in self.coils:
            # Concat coils and bounds
            for a_bound in coil.coef_channel_minmax:
                bounds.append(a_bound)

        return bounds

    def _check_sizing(self, mask):
        """
        Helper function to check array sizing

        Args:
            mask (np.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                                  unshimmed
        """

        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (dim1, dim2, dim3)")
        if mask.shape != self.unshimmed.shape:
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the unshimmed image"
                             f" with shape: {self.unshimmed.shape}")
