#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class Coil(object):
    """
    Coil profile object that stores coil profiles and there constraints

    Attributes:
        x (int): Amount of pixels in the X direction
        y (int): Amount of pixels in the Y direction
        z (int): Amount of pixels in the Z direction
        n (int): Amount of channels in the coil profile
        profiles (numpy.ndarray): (X, Y, Z, N) 4d array of N 3d coil profiles
        coef_sum_max (float): Contains the maximum value for the sum of the coefficients
        coef_channel_minmax (list): Contains the maximum coefficient for each channel
    """

    def __init__(self, profiles, constraints):
        """

        Args:
            profiles (np.ndarray): Coil profile (x, y, z, channel) 4d array of N 3d coil profiles
            constraints (dict): dict containing the constraints for the coil profiles. Required keys:
                coef_sum_max (float): Contains the maximum value for the sum of the coefficients
                coef_channel_max (list): List of ``(min, max)`` pairs for each coil channels. None
                                         is used to specify no bound.
                Example:
                    constraints = {
                        "coef_sum_max": 40,
                            # 8 channel coil
                        "coef_channel_minmax": [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]
                    }
        """

        self.x = self.y = self.z = self.n = -1
        self.profiles = profiles

        self.coef_channel_minmax = self.coef_sum_max = -1
        self.load_constraints(constraints)

    @property
    def profiles(self):
        return self._profiles

    @profiles.setter
    def profiles(self, profiles):
        if profiles.ndim != 4:
            raise ValueError(f"Coil profile has {profiles.ndim} dimensions, expected 4 (X, Y, Z, N)")
        self.x, self.y, self.z, self.n = profiles.shape
        self._profiles = profiles

    def load_constraints(self, constraints):
        """Loads the constraints named in required_constraints as attribute to this class"""

        required_constraints = [
            "coef_channel_minmax",
            "coef_sum_max"
        ]

        for key_name in required_constraints:
            if key_name in constraints:

                if key_name == "coef_channel_max":
                    if len(constraints[key_name]) != self.n:
                        raise ValueError(f"length of 'coef_channel_max' must be the same as the number of channels: "
                                         f"{self.n}")

                setattr(self, key_name, constraints[key_name])
            else:
                raise KeyError(f"Missing required constraint: {key_name}")
