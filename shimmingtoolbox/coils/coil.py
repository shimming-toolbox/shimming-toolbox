#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple

from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.coils.coordinates import generate_meshgrid

required_constraints = [
    "name",
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
        coef_channel_minmax (list): List of ``(min, max)`` pairs for each coil channels. (None, None) is
                                    used to specify no bounds.
        name (str): Name of the coil.
    """

    def __init__(self, profile, affine, constraints):
        """ Initialize Coil

        Args:
            profile (np.ndarray): Coil profile (dim1, dim2, dim3, channels) 4d array of N 3d coil profiles
            affine (np.ndarray): 4x4 array containing the qform affine transformation for the coil profiles
            constraints (dict): dict containing the constraints for the coil profiles. Required keys:

                * name (str): Name of the coil.
                * coef_sum_max (float): Contains the maximum value for the sum of the coefficients. None is used to
                  specify no bounds
                * coef_channel_max (list): List of ``(min, max)`` pairs for each coil channels. (None, None) is
                  used to specify no bounds.

        Examples:

            ::

                # Example of constraints
                constraints = {
                    'name': "dummy coil",
                    'coef_sum_max': 10,
                    # 8 channel coil
                    'coef_channel_minmax': [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)],
                }
        """

        self.dim = (np.nan,) * 4
        self.profile = profile
        self.required_constraints = required_constraints

        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")
        self.affine = affine

        self.name = ""
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

                if key_name == "coef_channel_minmax":
                    if len(constraints["coef_channel_minmax"]) != self.dim[3]:
                        raise ValueError(f"length of 'coef_channel_max' must be the same as the number of channels: "
                                         f"{self.dim[3]}")

                    for i_channel in range(self.dim[3]):
                        if constraints["coef_channel_minmax"][i_channel] is None:
                            constraints["coef_channel_minmax"][i_channel] = (-np.inf, np.inf)
                        if constraints["coef_channel_minmax"][i_channel][0] is None:
                            constraints["coef_channel_minmax"][i_channel] = \
                                (-np.inf, constraints["coef_channel_minmax"][i_channel][1])
                        if constraints["coef_channel_minmax"][i_channel][1] is None:
                            constraints["coef_channel_minmax"][i_channel] = \
                                (constraints["coef_channel_minmax"][i_channel][0], np.inf)

                if key_name == "coef_sum_max":
                    if constraints["coef_sum_max"] is None:
                        constraints["coef_sum_max"] = np.inf

                setattr(self, key_name, constraints[key_name])
            else:
                raise KeyError(f"Missing required constraint: {key_name}")


class ScannerCoil(Coil):
    """Coil class for scanner coils as they require extra arguments"""
    def __init__(self, coord_system, dim_volume, affine, constraints, order):

        self.order = order
        self.coord_system = coord_system
        self.affine = affine

        # Create the spherical harmonics with the correct order, dim and affine
        # Todo: add coord system
        sph_coil_profile = self._create_coil_profile(dim_volume)
        # Restricts the constraints to the specified order
        sph_constraints = self._restrict_constraints(constraints)

        super().__init__(sph_coil_profile, affine, sph_constraints)

    def _restrict_constraints(self, in_contraints):
        # Restrict constraint coefficient size/bounds depending on the order
        out_constraints = in_contraints
        if self.order == 0:
            # f0 --> [1]
            out_constraints['coef_channel_minmax'] = in_contraints['coef_channel_minmax'][:1]
        elif self.order == 1:
            # f0, ch1, ch2, ch3 -- > [4]
            # Order 1 only requires the first 3 channels + Tx
            out_constraints['coef_channel_minmax'] = in_contraints['coef_channel_minmax'][:4]
        elif self.order == 2:
            # f0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8 -- > [9]
            # Order 2 requires 8 channels + Tx
            out_constraints['coef_channel_minmax'] = in_contraints['coef_channel_minmax'][:9]

        return out_constraints

    def _create_coil_profile(self, dim):
        # Define profile for Tx (constant volume)
        profile_order_0 = np.ones(dim)

        # define the coil profiles
        if self.order == 0:
            # f0 --> [1]
            sph_coil_profile = profile_order_0[..., np.newaxis]
        else:
            # f0, orders
            mesh1, mesh2, mesh3 = generate_meshgrid(dim, self.affine)
            profile_orders = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(range(1, self.order + 1)))
            sph_coil_profile = np.concatenate((profile_order_0[..., np.newaxis], profile_orders), axis=3)

        return sph_coil_profile


def convert_to_mp(shim_setting, manufacturers_model_name):
    """ Converts the ShimSettings tag from the json BIDS sidecar to the scanner units.
        (i.e. For the Prisma fit DAC --> uT/m, uT/m^2 (1st order, 2nd order))

    Args:
        shim_setting (list): List of coefficients. Found in the json BIDS sidecar under 'ShimSetting'.
        manufacturers_model_name (str): Name of the model of the scanner. Found in the json BIDS sidecar under
                                        ManufacturersModelName'. Supported names: 'Prisma_fit'.

    Returns:
        list: Coefficients with units converted.
    """

    if manufacturers_model_name == "Prisma_fit":
        # One can use the Siemens commandline AdjValidate tool to get all the values below:
        max_current_mp = np.array([2300, 2300, 2300, 4959.01, 3551.29, 3503.299, 3551.29, 3487.302])
        max_current_dcm = np.array([14436, 14265, 14045, 9998, 9998, 9998, 9998, 9998])

        shim_setting = np.array(shim_setting) * max_current_mp / max_current_dcm

        if np.any(np.abs(shim_setting) > max_current_mp):
            raise ValueError("Multipole values exceed known system limits.")

    else:
        raise NotImplementedError("Manufacturer model not recognized, could not convert units")

    return list(shim_setting)
