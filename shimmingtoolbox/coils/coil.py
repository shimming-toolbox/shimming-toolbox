#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import logging

import numpy as np
from typing import Tuple

from shimmingtoolbox.coils.spher_harm_basis import siemens_basis, ge_basis, philips_basis, SHIM_CS
from shimmingtoolbox.coils.coordinates import generate_meshgrid

logger = logging.getLogger(__name__)

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
        coef_channel_minmax (dict): Dict of ``(min, max)`` pairs for each coil channels. (None, None) is
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

    def load_constraints(self, constraints: dict):
        """Loads the constraints named in required_constraints as attribute to this class"""
        # global `required_constraints`
        for key_name in required_constraints:
            if key_name in constraints:
                if key_name == "coef_channel_minmax":
                    if sum([len(constraints["coef_channel_minmax"][key]) for key in
                            constraints["coef_channel_minmax"]]) != self.dim[3]:
                        raise ValueError(f"length of 'coef_channel_max' must be the same as the number of channels: "
                                         f"{self.dim[3]} {sum([len(constraints['coef_channel_minmax'][key]) for key in constraints['coef_channel_minmax']])}")

                    for key in constraints["coef_channel_minmax"]:
                        for i in range(len(constraints["coef_channel_minmax"][key])):
                            if constraints["coef_channel_minmax"][key][i] is None:
                                constraints["coef_channel_minmax"][key][i] = (-np.inf, np.inf)
                            if constraints["coef_channel_minmax"][key][i][0] is None:
                                constraints["coef_channel_minmax"][key][i] = \
                                    (-np.inf, constraints["coef_channel_minmax"][key][i][1])
                            if constraints["coef_channel_minmax"][key][i][1] is None:
                                constraints["coef_channel_minmax"][key][i] = \
                                    (constraints["coef_channel_minmax"][key][i][0], np.inf)

                if key_name == "coef_sum_max":
                    if constraints["coef_sum_max"] is None:
                        constraints["coef_sum_max"] = np.inf

                setattr(self, key_name, constraints[key_name])
            else:
                raise KeyError(f"Missing required constraint: {key_name}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        return self.name == __value.name


class ScannerCoil(Coil):
    """Coil class for scanner coils as they require extra arguments"""

    def __init__(self, dim_volume, affine, constraints, orders, manufacturer=""):

        self.orders = orders

        manufacturer = manufacturer.upper()
        if manufacturer in SHIM_CS:
            self.coord_system = SHIM_CS[manufacturer.upper()]
        else:
            logger.warning(f"Unknown manufacturer {manufacturer}, assuming RAS")
            self.coord_system = 'RAS'

        self.affine = affine

        # Create the spherical harmonics with the correct order, dim and affine
        sph_coil_profile = self._create_coil_profile(dim_volume, manufacturer)
        # Restricts the constraints to the specified order
        constraints['coef_channel_minmax'] = restrict_sph_constraints(constraints['coef_channel_minmax'], self.orders)

        super().__init__(sph_coil_profile, affine, constraints)

    def _create_coil_profile(self, dim, manufacturer=None):
        # Define profile for Tx (constant volume)
        if 0 in self.orders:
            profile_order_0 = -np.ones(dim)
        else:
            profile_order_0 = None
        # define the coil profiles
        if self.orders == [0]:
            # f0 --> [1]
            sph_coil_profile = profile_order_0[..., np.newaxis]
        else:
            # f0, orders
            temp_orders = [order for order in self.orders if order != 0]
            mesh1, mesh2, mesh3 = generate_meshgrid(dim, self.affine)

            if manufacturer == 'SIEMENS':
                profile_orders = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(temp_orders),
                                               shim_cs=self.coord_system)
            elif manufacturer == 'GE':
                profile_orders = ge_basis(mesh1, mesh2, mesh3, orders=tuple(temp_orders),
                                          shim_cs=self.coord_system)
            elif manufacturer == 'PHILIPS':
                profile_orders = philips_basis(mesh1, mesh2, mesh3, orders=tuple(temp_orders),
                                               shim_cs=self.coord_system)
            else:
                logger.warning(f"{manufacturer} manufacturer not implemented. Outputting in Hz, uT/m, uT/m^2 for order "
                               f"0, 1 and 2 respectively")
                profile_orders = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(temp_orders),
                                               shim_cs=self.coord_system)
            if profile_order_0 is None:
                sph_coil_profile = profile_orders
            else:
                sph_coil_profile = np.concatenate((profile_order_0[..., np.newaxis], profile_orders), axis=3)

        return sph_coil_profile

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)


def get_scanner_constraints(manufacturers_model_name, orders):
    """ Returns the scanner spherical harmonics constraints depending on the manufacturer's model name and required
        order

    Args:
        manufacturers_model_name (str): Name of the scanner
        orders (list): List of all orders of the shim system to be used

    Returns:
        dict: The constraints including the scanner name, bounds and the maximum sum of currents.
    """

    if manufacturers_model_name == "Prisma_fit":
        constraints = {
            "name": "Prisma_fit",
            "coef_channel_minmax": {"0": [], "1": [], "2": []},
            "coef_sum_max": None
        }
        if 0 in orders:
            constraints["coef_channel_minmax"]["0"].append([123100100, 123265000])
        if 1 in orders:
            for _ in range(3):
                constraints["coef_channel_minmax"]["1"].append([-2300, 2300])
        if 2 in orders:
            constraints["coef_channel_minmax"]["2"].extend([[-4959.01, 4959.01],
                                                            [-3551.29, 3551.29],
                                                            [-3503.299, 3503.299],
                                                            [-3551.29, 3551.29],
                                                            [-3487.302, 3487.302]])

    elif manufacturers_model_name == "Investigational_Device_7T":
        constraints = {
            "name": "Investigational_Device_7T",
            "coef_channel_minmax": {"0": [], "1": [], "2": []},
            "coef_sum_max": None
        }
        if 0 in orders:
            pass
            # todo: f0 min and max is wrong
        constraints["coef_channel_minmax"]["0"].append([None, None])
        if 1 in orders:
            for _ in range(3):
                constraints["coef_channel_minmax"]["1"].append([-5000, 5000])
        if 2 in orders:
            constraints["coef_channel_minmax"]["2"].extend([[-1839.63, 1839.63],
                                                            [-791.84, 791.84],
                                                            [-791.84, 791.84],
                                                            [-615.87, 615.87],
                                                            [-615.87, 615.87]])
    else:
        logger.warning(f"Scanner: {manufacturers_model_name} constraints not yet implemented, constraints might not be "
                       "respected.")
        constraints = {
            "name": "Unknown",
            "coef_channel_minmax": {"0": [], "1": [], "2": []},
            "coef_sum_max": None
        }

        if 0 in orders:
            constraints["coef_channel_minmax"]["0"] = [[None, None]]
        if 1 in orders:
            constraints["coef_channel_minmax"]["1"] = [[None, None] for _ in range(3)]
        if 2 in orders:
            constraints["coef_channel_minmax"]["2"] = [[None, None] for _ in range(5)]

    return constraints


def restrict_sph_constraints(bounds: dict, orders):
    # ! Modify description if everything works
    """ Select bounds according to the order specified

    Args:
        bounds (dict): Dictionary containing the min and max currents for multiple spherical harmonics
                       orders
        orders (list): Lsit of all spherical harmonics orders to be used

    Returns:
        dict: Dictionary with the bounds of all specified orders
    """
    minmax_out = {}
    if 0 in orders:
        # f0 --> [1]
        minmax_out["0"] = bounds["0"]
    if 1 in orders:
        # f0, ch1, ch2, ch3 -- > [4]
        minmax_out["1"] = bounds["1"]
    if 2 in orders:
        # f0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8 -- > [9]
        minmax_out["2"] = bounds["2"]
    if minmax_out == {}:
        raise NotImplementedError("Order must be between 0 and 2")

    return minmax_out
