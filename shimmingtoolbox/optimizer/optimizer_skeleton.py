#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import logging


class Optimizer(object):
"""
Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask
"""
    def __init__(self, coil_profiles=None):
    """
    Args:
        coil_profiles (np.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
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
        coil_profiles (np.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
    """
        self._error_if(coil_profiles.ndim != 4,
                       f"Coil profile has {coil_profiles.ndim} dimensions, expected 4 (X, Y, Z, N)")
        self.X, self.Y, self.Z, self.N = coil_profiles.shape
        self.coils = coil_profiles

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0)):
    """
    Optimize unshimmed volume by varying current to each channel

    Args:
        unshimmed (np.ndarray): (X, Y, Z) 3d array of unshimmed volume
        mask (np.ndarray of ints): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
        mask_origin (3-tuple): Origin of mask if mask volume does not cover unshimmed volume
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
                           f"Mask (shape: {mask.shape}, origin: {mask_origin}) goes out of bounds (coil shape: {(self.X, self.Y, self.Z)}")

        # Set up output currents and optimize
        output = np.zeros(self.N)

        return output

    # TODO: refactor to raise errors instead of assert
    def _error_if(self, err_condition, message):
    """
    Helper function throwing errors

    Args:
        err_condition (bool): Condition to throw error on
        message (string): Message to log and throw
    """
        if err_condition: self.logger.error(message)
        assert not err_condition, message
