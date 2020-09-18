#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import logging


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask

    Attributes:
        X (int): Amount of pixels in the X direction
        Y (int): Amount of pixels in the Y direction
        Z (int): Amount of pixels in the Z direction
        N (int): Amount of channels in the coil profile
        coils (numpy.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
    """

    def __init__(self, coil_profiles=None):
        """
        Initializes X, Y, Z, N and coils according to input coil_profiles

        Args:
            coil_profiles (numpy.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
        """
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        # Load coil profiles (X, Y, Z, N) if given
        if coil_profiles is None:
            self.X = 0
            self.Y = 0
            self.Z = 0
            self.N = 0
            self.coils = None
        else:
            self.load_coil_profiles(coil_profiles)

    # Load coil profiles and check dimensions
    def load_coil_profiles(self, coil_profiles):
        """
        Load new coil profiles into Optimizer

        Args:
            coil_profiles (numpy.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
        """
        self._error_if(coil_profiles.ndim != 4,
                       f"Coil profile has {coil_profiles.ndim} dimensions, expected 4 (X, Y, Z, N)")
        self.X, self.Y, self.Z, self.N = coil_profiles.shape
        self.coils = coil_profiles

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0), bounds=None):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): (X, Y, Z) 3d array of unshimmed volume
            mask (numpy.ndarray): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
            mask_origin (tuple): Origin of mask if mask volume does not cover unshimmed volume
            bounds (list): List of ``(min, max)`` pairs for each coil channels. None
               is used to specify no bound.
        """

        # Check for sizing errors
        self._error_if(self.coils is None, "No loaded coil profiles!")
        self._error_if(unshimmed.ndim != 3,
                       f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (X, Y, Z)")
        self._error_if(mask.ndim != 3, f"Mask has {mask.ndim} dimensions, expected 3 (X, Y, Z)")
        self._error_if(unshimmed.shape != (self.X, self.Y, self.Z),
                       f"XYZ mismatch -- Coils: {self.coils.shape}, Unshimmed: {unshimmed.shape}")
        for i in range(3):
            self._error_if(mask.shape[i] + mask_origin[i] > (self.X, self.Y, self.Z)[i],
                           f"Mask (shape: {mask.shape}, origin: {mask_origin}) goes out of bounds (coil shape: "
                           f"{(self.X, self.Y, self.Z)}")
        self._error_if(len(bounds) != self.N and bounds is not None, f"Bounds should have the same number of (min, max)"
                                                                     f" tuples as coil as channels")

        # Set up output currents and optimize
        output = np.zeros(self.N)

        # Simple pseudo-inverse optimization
        profile_mat = np.reshape(np.transpose(self.coils, axes=(3, 0, 1, 2)), (self.N, -1)).T # V x N
        unshimmed_vec = np.reshape(unshimmed, (self.N,)) # V

        output = -1 * scipy.linalg.pinv(profile_mat) @ unshimmed_vec # N x V @ V

        return output

    def _error_if(self, err_condition, message):
        """
        Helper function throwing errors

        Args:
            err_condition (bool): Condition to throw error on
            message (string): Message to log and throw
        """
        if err_condition:
            self.logger.error(message)
            raise RuntimeError(message)
