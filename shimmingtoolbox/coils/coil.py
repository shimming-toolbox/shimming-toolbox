#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple

required_constraints = [
    "coef_channel_minmax",
    "coef_sum_max"
]


class Coil(object):
    """
    Coil profile object that stores coil profiles and there constraints

    Attributes:
        dim (Tuple[int]): Dimension along specific axis. dim: 0,1,2 are spatial axes, while dim: 3 corresponds to the
                          coil channel.
        profile (np.ndarray): (dim1, dim2, dim3, channels) 4d array of N 3d coil profiles
        affine (np.ndarray): 4x4 array containing the affine transformation associated with the NIfTI file of the coil
                             profile. This transformation relates to the physical coordinates of the scanner (qform).
        required_constraints (list): List containing the required keys for ``constraints``
        coef_sum_max (float): Contains the maximum value for the sum of the coefficients
        coef_channel_minmax (list): Contains the maximum coefficient for each channel
    """

    def __init__(self, profile, affine, constraints):
        """ Initialize Coil

        Args:
            profile (np.ndarray): Coil profile (dim1, dim2, dim3, channels) 4d array of N 3d coil profiles
            affine (np.ndarray): 4x4 array containing the qform affine transformation for the coil profiles
            constraints (dict): dict containing the constraints for the coil profiles. Required keys:

                * coef_sum_max (float): Contains the maximum value for the sum of the coefficients
                * coef_channel_max (list): List of ``(min, max)`` pairs for each coil channels. None is used to specify
                  no bound.

        Examples:

            ::

                # Example of constraints
                constraints = {
                    'coef_sum_max': 40,
                    # 8 channel coil
                    'coef_channel_minmax': [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]
                }
        """

        self.dim = (np.nan,) * 4
        self.profile = profile
        self.required_constraints = required_constraints

        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")
        self.affine = affine

        self.coef_channel_minmax = self.coef_sum_max = -1
        self.load_constraints(constraints)

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, profile):
        if profile.ndim != 4:
            raise ValueError(f"Coil profile has {profile.ndim} dimensions, expected 4 (dim1, dim2, dim3, channel)")
        self.dim = profile.shape
        self._profile = profile

    def load_constraints(self, constraints):
        """Loads the constraints named in required_constraints as attribute to this class"""

        # global `required_constraints`
        for key_name in required_constraints:
            if key_name in constraints:

                if key_name == "coef_channel_max":
                    if len(constraints[key_name]) != self.dim[3]:
                        raise ValueError(f"length of 'coef_channel_max' must be the same as the number of channels: "
                                         f"{self.dim[3]}")

                setattr(self, key_name, constraints[key_name])
            else:
                raise KeyError(f"Missing required constraint: {key_name}")
