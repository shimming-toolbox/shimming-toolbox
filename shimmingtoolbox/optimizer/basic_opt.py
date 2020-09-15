#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from shimmingtoolbox.optimizer.optimizer_skeleton import Optimizer


class BasicOptimizer(Optimizer):

    def _objective(self, currents, masked_unshimmed, masked_coils):
        """
        TODO:
        Args:
            currents:
            masked_unshimmed:
            masked_coils:

        Returns:

        """
        shimmed = masked_unshimmed + np.sum(masked_coils * currents, axis=3, keepdims=False)
        objective = np.std(shimmed) + np.sum(currents)/100000
        return objective

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0)):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): 3D B0 map
            mask (numpy.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).
            mask_origin (tuple): Mask origin if mask volume does not cover unshimmed volume

        Returns:
            numpy.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function
                           (coils.size)
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

        # Set up mask
        full_mask = np.zeros((self.X, self.Y, self.Z))
        full_mask[mask_origin[0]:mask_origin[0] + mask.shape[0], mask_origin[1]:mask_origin[1] + mask.shape[1], mask_origin[2]:mask_origin[2] + mask.shape[2]] = mask
        full_mask = np.where(full_mask != 0, 1, 0)
        
        masked_unshimmed = unshimmed * full_mask
        masked_coils = self.coils * full_mask.reshape(full_mask.shape + (1,))

        # Set up output currents and optimize
        currents = np.zeros(self.N)
        # TODO: min and max coef are currently arbitrary, put as inputs?
        max_coef = 5000
        min_coef = -5000
        bounds = []
        for i_coils in range(self.N):
            bounds.append((min_coef, max_coef))

        currents = opt.minimize(self._objective, currents, args=(masked_unshimmed, masked_coils), bounds=bounds).x

        return currents

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
