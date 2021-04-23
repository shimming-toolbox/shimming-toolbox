#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import logging

from shimmingtoolbox.coils.coil import Coil


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
    Use optimize(args) to optimize a given mask.
    For basic optimizer, uses unbounded pseudo-inverse.

    Attributes:
        coil (Coil): Coil object containing the coil profiles and related constraints
    """

    def __init__(self, coil: Coil):
        """
        Initializes X, Y, Z, N and coils according to input coil_profiles

        Args:
            coil (Coil): Coil object containing the coil profiles and related constraints
        """
        # TODO: Change to a list of coils
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        self.coil = coil

    def optimize(self, unshimmed, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): (X, Y, Z) 3d array of unshimmed volume
            mask (numpy.ndarray): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
        """
        # Check for sizing errors
        self._check_sizing(unshimmed, mask)

        # Set up output currents and optimize
        # output = np.zeros(self.coil.n)

        mask_vec = mask.reshape((-1,))

        # Simple pseudo-inverse optimization
        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        coil_mat = np.reshape(np.transpose(self.coil.profiles, axes=(3, 0, 1, 2)),
                              (self.coil.n, -1)).T[mask_vec != 0, :]  # masked points x N
        unshimmed_vec = np.reshape(unshimmed, (-1,))[mask_vec != 0]  # mV'

        output = -1 * scipy.linalg.pinv(coil_mat) @ unshimmed_vec  # N x mV' @ mV'

        return output

    def _check_sizing(self, unshimmed, mask):
        """
        Helper function to check array sizing

        Args:
            unshimmed (numpy.ndarray): (X, Y, Z) 3d array of unshimmed volume
            mask (numpy.ndarray): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
        """

        if unshimmed.ndim != 3:
            raise ValueError(f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (X, Y, Z)")
        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (X, Y, Z)")
        if unshimmed.shape != (self.coil.x, self.coil.y, self.coil.z):
            raise ValueError("XYZ mismatch -- Coils: {self.coil.profiles.shape}, Unshimmed: {unshimmed.shape}")

        if mask.shape != (self.coil.x, self.coil.y, self.coil.z):
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the coil profiles with"
                             f" shape: {self.coil.profiles.shape}")
