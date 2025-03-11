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
            affine (np.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
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
            raise ValueError(f"Size of merged bounds: must match the number of total "
                             f"channel: {len(self.merged_bounds)} not {len(merged_bounds)}")

        logger.debug(f"Merged bounds: {merged_bounds}")
        self.merged_bounds = merged_bounds

    def optimize(self, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (np.ndarray): 3d array marking volume for optimization. Must be the same shape as unshimmed

        Returns:
            profiles (np.ndarray): Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """
        weighted_coil_mat, weighted_unshimmed_vec = self.get_weighted_coil_mat_and_unshimmed(mask)

        # Compute the pseudo-inverse of the coil matrix to get the desired coil profiles
        # dimensions : pinv(1, N) * scalar --> (N, 1) * scalar --> (N, 1)
        profiles = -1 * scipy.linalg.pinv(weighted_coil_mat[:, np.newaxis]) * weighted_unshimmed_vec

        return profiles

    def get_coil_mat_and_unshimmed(self, mask):
        """
        Returns the coil matrix, and the unshimmed vector used for the optimization

        Args:
            mask (np.ndarray): 3d array marking volume for optimization. Must be the same shape as unshimmed

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 2D flattened array (point, channel) of masked coils
                              (axis 0 must align with unshimmed_vec)
                * np.ndarray: 1D flattened array (point) of the masked unshimmed map
        """
        # Check for sizing errors
        self._check_sizing(mask)
        # Reshape mask to 1D
        mask_vec = mask.reshape((-1,))
        # Get indices of non-zero mask values
        masked_points_indices = np.where(mask_vec != 0.0)

        # Define number of coil profiles (channels)
        n_channels = self.merged_coils.shape[3]
        # Transpose coil profile : (X, Y, Z, N) --> (N, X, Y, Z) or (N, [mask.shape])
        merged_coils_transposed = np.transpose(self.merged_coils, axes=(3, 0, 1, 2))
        # Reshape coil profile : (N, X, Y, Z) --> (N, X * Y * Z) or (N, mask.size)
        merged_coils_reshaped = np.reshape(merged_coils_transposed, (n_channels, -1))

        # Extract the masked coil matrix
        # dimensions : (N, mask.size) --> (mask.size, N) --> (masked values, N) or (mV', N)
        coil_mat = merged_coils_reshaped[:, masked_points_indices[0]].T
        # Extract the unshimmed vector
        # dimensions : (mV',)
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[masked_points_indices[0]]

        return coil_mat, unshimmed_vec

    def get_weighted_coil_mat_and_unshimmed(self, mask):
        """
        Returns the weighted coil matrix, and the weighted unshimmed vector used for the optimization

        Args:
            mask (np.ndarray): 3d array marking volume for optimization. Must be the same shape as unshimmed

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 2D flattened array (point, channel) of masked coils
                              (axis 0 must align with unshimmed_vec)
                * np.ndarray: 1D flattened array (point) of the masked unshimmed map
        """
        # Check for sizing errors
        self._check_sizing(mask)
        # Convert mask to float
        if mask.dtype != float:
            mask = mask.astype(float)
        # Reshape mask to 1D
        mask_vec = mask.reshape((-1,))
        # Get indices of non-zero mask values
        masked_points_indices = np.where(mask_vec != 0.0)
        # Get the non-zero mask coefficient values
        mask_coefficients = np.array(mask_vec[masked_points_indices[0]])
        # Create a vector with the square root of the coefficient values
        weight_vec = np.sqrt(mask_coefficients) # dimensions : (mV', 1)

        # Define number of coil profiles (channels)
        n_channels = self.merged_coils.shape[3]
        # Transpose coil profile : (X, Y, Z, N) --> (N, X, Y, Z) or (N, [mask.shape])
        merged_coils_transposed = np.transpose(self.merged_coils, axes=(3, 0, 1, 2))
        # Reshape coil profile : (N, X, Y, Z) --> (N, X * Y * Z) or (N, mask.size)
        merged_coils_reshaped = np.reshape(merged_coils_transposed, (n_channels, -1))

        # Extract the masked coil matrix
        # dimensions : (N, mask.size) --> (mask.size, N) --> (masked values, N) or (mV', N)
        coil_mat = merged_coils_reshaped[:, masked_points_indices[0]].T
        # Extract the unshimmed vector
        # dimensions : (mV', 1)
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[masked_points_indices[0]]

        # Apply weights to the coil matrix and unshimmed vector
        weighted_coil_mat = weight_vec.T @ coil_mat # dimensions : (1, mV') @ (mV', N) --> (1, N)
        weighted_unshimmed_vec = weight_vec.T @ unshimmed_vec # dimensions : (1, mV') @ (mV', 1) --> scalar

        return weighted_coil_mat, weighted_unshimmed_vec

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
            for key in coil.coef_channel_minmax:
                for a_bound in coil.coef_channel_minmax[key]:
                    bounds.append(a_bound)

        logger.debug(f"Merged bounds: {bounds}")
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
