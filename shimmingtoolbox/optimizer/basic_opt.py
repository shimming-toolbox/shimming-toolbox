#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from shimmingtoolbox.optimizer.optimizer_skeleton import Optimizer
import scipy.optimize as opt


class BasicOptimizer(Optimizer):

    def _objective(self, currents, masked_unshimmed, masked_coils):
        shimmed = masked_unshimmed + np.sum(masked_coils * currents, axis=3, keepdims=False)
        objective = np.std(shimmed) + np.sum(currents)/100000
        return objective

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0)):

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
        max_coef = 1000
        min_coef = -1000
        bounds = [(min_coef, max_coef), (min_coef, max_coef), (min_coef, max_coef), (min_coef, max_coef),
                  (min_coef, max_coef), (min_coef, max_coef), (min_coef, max_coef), (min_coef, max_coef)]
        currents = opt.minimize(self._objective, currents, args=(masked_unshimmed, masked_coils), bounds=bounds).x

        print(currents)
        return currents

    # For crashing and logging errors -- needs refactoring to raise instead of assert
    def _error_if(self, err_condition, message):
        if err_condition: self.logger.error(message)
        assert not err_condition, message
