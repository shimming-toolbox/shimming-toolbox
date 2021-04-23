#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import logging
from typing import List

from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
    Use optimize(args) to optimize a given mask.
    For basic optimizer, uses unbounded pseudo-inverse.

    Attributes:
        coil (Coil): Coil object containing the coil profiles and related constraints
    """

    def __init__(self, list_coil: ListCoil):
        """
        Initializes X, Y, Z, N and coils according to input coil_profiles

        Args:
            coil (ListCoil): List of Coil objects containing the coil profiles and related constraints
        """
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        self.list_coil = list_coil

    def optimize(self, unshimmed, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): (X, Y, Z) 3d array of unshimmed volume
            mask (numpy.ndarray): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
        """
        # Check for sizing errors
        self._check_sizing(unshimmed, mask)

        # Define coil profiles
        coil_profiles, _ = self.concat_coils()

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

    def concat_coils(self):
        """ Uses the list of coil profiles to return a concatenated list of coil profiles"""
        coil_profiles_list = []
        bounds = []
        for a_coil in self.list_coil:
            coil_profiles_list.append(a_coil.profiles)
            for a_bound in a_coil.coef_channel_minmax:
                bounds.append(a_bound)

        coil_profiles = np.concatenate(coil_profiles_list, axis=3)

        return coil_profiles, bounds

    def _check_sizing(self, unshimmed, mask):
        """
        Helper function to check array sizing

        Args:
            unshimmed (numpy.ndarray): (X, Y, Z) 3d array of unshimmed volume
            mask (numpy.ndarray): (X, Y, Z) 3d array of integers marking volume for optimization -- 0 indicates unused
        """
        # Check dimensions of coil profiles
        coil_shape = self.list_coil[0].profiles.shape
        for i in range(1, len(self.list_coil)):
            if coil_shape != self.list_coil[i].profiles.shape:
                raise ValueError("Dimension of coil profiles are not the same")

        if unshimmed.ndim != 3:
            raise ValueError(f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (X, Y, Z)")
        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (X, Y, Z)")
        if unshimmed.shape != (coil_shape[0], coil_shape[1], coil_shape[2]):
            raise ValueError("XYZ mismatch -- Coils: {self.coil.profiles.shape}, Unshimmed: {unshimmed.shape}")

        if mask.shape != (coil_shape[0], coil_shape[1], coil_shape[2]):
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the coil profiles with"
                             f" shape: {coil_shape}")
