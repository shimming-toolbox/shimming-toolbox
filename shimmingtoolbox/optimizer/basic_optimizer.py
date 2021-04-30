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
    """

    def __init__(self, coils: ListCoil):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
        """
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        self.coils = coils

    def optimize(self, unshimmed, affine, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            mask (numpy.ndarray): 3d array of integers marking volume for optimization -- 0 indicates unused. Nust be
                                  the same shape as unshimmed
        """
        # Check for sizing errors
        self._check_sizing(unshimmed, affine, mask)

        # Define coil profiles
        coil_profiles, _ = self.merge_coils(unshimmed, affine)

        # Optimize
        mask_vec = mask.reshape((-1,))

        # Simple pseudo-inverse optimization
        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        coil_mat = np.reshape(np.transpose(coil_profiles, axes=(3, 0, 1, 2)),
                              (coil_profiles.shape[3], -1)).T[mask_vec != 0, :]  # masked points x N
        unshimmed_vec = np.reshape(unshimmed, (-1,))[mask_vec != 0]  # mV'

        output = -1 * scipy.linalg.pinv(coil_mat) @ unshimmed_vec  # N x mV' @ mV'

        return output

    def merge_coils(self, unshimmed, affine):
        """
        Uses the list of coil profiles to return a resampled concatenated list of coil profiles matching the
        unshimmed image. Bounds are also concatenated and returned.
        """

        coil_profiles_list = []
        bounds = []

        # Define the nibabel unshimmed array
        nii_unshimmed = nib.Nifti1Image(unshimmed, affine)

        for a_coil in self.coils:
            nii_coil = nib.Nifti1Image(a_coil.profile, a_coil.affine)

            # Resample a coil on the unshimmed image
            resampled_coil = resample_from_to(nii_coil, nii_unshimmed).get_fdata()

            # Concat coils and bounds
            coil_profiles_list.append(resampled_coil)
            for a_bound in a_coil.coef_channel_minmax:
                bounds.append(a_bound)

        coil_profiles = np.concatenate(coil_profiles_list, axis=3)

        return coil_profiles, bounds

    def _check_sizing(self, unshimmed, affine, mask):
        """
        Helper function to check array sizing

        Args:
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            mask (numpy.ndarray): 3d array of integers marking volume for optimization -- 0 indicates unused. Must be
                                  the same shape as unshimmed
        """

        # Check dimenssions of affine
        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")

        # Check dimensions of unshimmed map, mask annd make sure they are the same shape
        if unshimmed.ndim != 3:
            raise ValueError(f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (dim1, dim2, dim3)")
        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (dim1, dim2, dim3)")
        if mask.shape != unshimmed.shape:
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the unshimmed image"
                             f" with shape: {unshimmed.shape}")
