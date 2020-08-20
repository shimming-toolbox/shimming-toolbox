# coding: utf-8

from pathlib import Path
import shutil
import os
import numpy as np
import logging


class Optimizer(object):

    def __init__(self, coil_profiles=None):
        
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename = 'test_optimizer.log', filemode = 'w', level = logging.DEBUG)

        # Load coil profiles (N, X, Y, Z) if given
        if coil_profiles is None:
            self.N = 0
            self.X = 0
            self.Y = 0
            self.Z = 0
            self.coils = None
        else:
            self.load_coil_profiles(coil_profiles)

    # Load coil profiles and check dimensions
    def load_coil_profiles(self, coil_profiles):
        self._error_if(len(coil_profiles.shape) != 4, f"Coil profile has {len(coil_profiles.shape)} dimensions, expected 4 (N, X, Y, Z)")
        self.N, self.X, self.Y, self.Z = coil_profiles.shape
        self.coils = coil_profiles
    

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0)):

        # Check for sizing errors
        self._error_if(self.coils is None, "No loaded coil profiles!")
        self._error_if(len(unshimmed.shape) != 3, f"Unshimmed profile has {len(unshimmed.shape)} dimensions, expected 3 (X, Y, Z)")
        self._error_if(len(mask.shape) != 3, f"Mask has {len(mask.shape)} dimensions, expected 3 (X, Y, Z)")
        self._error_if(unshimmed.shape != (self.X, self.Y, self.Z), f"XYZ mismatch -- Coils: {self.coils.shape}, Unshimmed: {unshimmed.shape}")
        for i in range(3):
            self._error_if(mask.shape[i] + mask_origin[i] > (self.X, self.Y, self.Z)[i], 
                f"Mask (shape: {mask.shape}, origin: {mask_origin}) goes out of bounds (coil shape: {(self.X, self.Y, self.Z)}")
        
        # Set up output currents and optimize
        output = np.zeros(self.N)

        return output


    # For crashing and logging errors
    def _error_if(self, err_condition, message):
        if err_condition: self.logger.error(message)
        assert not err_condition, (message)
