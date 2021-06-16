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


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
    Use optimize(args) to optimize a given mask.
    For basic optimizer, uses unbounded pseudo-inverse.

    Attributes:
        coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
        unshimmed (numpy.ndarray): 3d array of unshimmed volume
        unshimmed_affine (numpy.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
        merged_coils (numpy.ndarray): 4d array containing all coil profiles resampled onto the target unshimmed array
                                      concatenated on the 4th dimension. See self.merge_coils() for more details
        merged_bounds (list): list of bounds corresponding to each merged coils: merged_bounds[3] is the (min, max)
                              bound for merged_coils[..., 3]
    """

    def __init__(self, coils: ListCoil, unshimmed, affine):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
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
        """ Set the unshimmed array to a new array. Resamples coil profiles accordingly.

        Args:
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine: (numpy.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
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

    def optimize(self, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (numpy.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                                  unshimmed

        Returns:
            numpy.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """
        # Check for sizing errors
        self._check_sizing(mask)

        # Optimize
        mask_vec = mask.reshape((-1,))

        # Simple pseudo-inverse optimization
        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        coil_mat = np.reshape(np.transpose(self.merged_coils, axes=(3, 0, 1, 2)),
                              (self.merged_coils.shape[3], -1)).T[mask_vec != 0, :]  # masked points x N
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_vec != 0]  # mV'

        output = -1 * scipy.linalg.pinv(coil_mat) @ unshimmed_vec  # N x mV' @ mV'

        return output

    def merge_coils(self, unshimmed, affine):
        """
        Uses the list of coil profiles to return a resampled concatenated list of coil profiles matching the
        unshimmed image. Bounds are also concatenated and returned.

        Args:
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        """

        coil_profiles_list = []
        bounds = []

        # Define the nibabel unshimmed array
        nii_unshimmed = nib.Nifti1Image(unshimmed, affine)

        for coil in self.coils:
            nii_coil = nib.Nifti1Image(coil.profile, coil.affine)

            # Resample a coil on the unshimmed image
            resampled_coil = resample_from_to(nii_coil, nii_unshimmed).get_fdata()

            # Concat coils and bounds
            coil_profiles_list.append(resampled_coil)
            for a_bound in coil.coef_channel_minmax:
                bounds.append(a_bound)

        coil_profiles = np.concatenate(coil_profiles_list, axis=3)

        return coil_profiles, bounds

    def _check_sizing(self, mask):
        """
        Helper function to check array sizing

        Args:
            mask (numpy.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                                  unshimmed
        """

        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (dim1, dim2, dim3)")
        if mask.shape != self.unshimmed.shape:
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the unshimmed image"
                             f" with shape: {self.unshimmed.shape}")
