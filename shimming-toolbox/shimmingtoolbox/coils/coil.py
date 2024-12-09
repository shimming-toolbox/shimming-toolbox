#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
from typing import Tuple

from shimmingtoolbox.coils.spher_harm_basis import (sh_basis, siemens_basis, ge_basis, philips_basis, SHIM_CS,
                                                    channels_per_order)
from shimmingtoolbox.coils.coordinates import generate_meshgrid

logger = logging.getLogger(__name__)

required_constraints = [
    "name",
    "coef_channel_minmax",
    "coef_sum_max"
]

SCANNER_CONSTRAINTS = {
    "Siemens": {
        "Prisma_fit": {
            "0": [[123100100, 123265000]],
            "1": [[-2300, 2300], [-2300, 2300], [-2300, 2300]],
            "2": [[-4959.01, 4959.01], [-3551.29, 3551.29], [-3503.299, 3503.299], [-3551.29, 3551.29],
                  [-3487.302, 3487.302]],
            "3": []
        },
        "Investigational_Device_7T": {
            "0": [[296490000, 297490000]],
            "1": [[-4999.976, 4999.976], [-4999.980, 4999.980], [-4999.957, 4999.957]],
            "2": [[-6163.2, 6163.2], [-2592.0, 2592.0], [-2592.0, 2592.0], [-2476.8, 2476.8], [-2476.8, 2476.8]],
            "3": []
        },
        "Terra": {
            "0": [[296760000, 297250000]],
            "1": [[-3000, 3000], [-3000, 3000], [-3000, 3000]],
            "2": [[-9360.0, 9360.0], [-4680.0, 4680.0], [-4620.0, 4620.0], [-4620.0, 4620.0], [-4560.0, 4560.0]],
            "3": [[-15232.0, 15232.0], [-14016.0, 14016.0], [-14016.0, 14016.0], [-14016.0, 14016.0]],
        }
    },
    "GE": {

    },
    "Philips": {

    }
}

# One can use the Siemens commandline AdjValidate tool to get all the values below
SCANNER_CONSTRAINTS_DAC = {
    "Siemens": {
        "Prisma_fit": {
            "1": [14436.0, 14265.0, 14045.0],
            "2": [9998.0] * 5,
            "3": []
        },
        "Investigational_Device_7T": {
            "1": [62479.0, 62264.0, 54082.0],
            "2": [18000.0] * 5,
            "3": []
        },
        "Terra": {
            "1": [17729.0, 18009.0, 17872.0],
            "2": [12500.0] * 5,
            "3": []
        }
    }
}


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
                                         f"{self.dim[3]}, currently:  {sum([len(constraints['coef_channel_minmax'][key]) for key in constraints['coef_channel_minmax']])}")

                    for key in constraints["coef_channel_minmax"]:
                        for i in range(len(constraints["coef_channel_minmax"][key])):
                            if constraints["coef_channel_minmax"][key][i] is None:
                                constraints["coef_channel_minmax"][key][i] = [-np.inf, np.inf]
                            if constraints["coef_channel_minmax"][key][i][0] is None:
                                constraints["coef_channel_minmax"][key][i] = \
                                    [-np.inf, constraints["coef_channel_minmax"][key][i][1]]
                            if constraints["coef_channel_minmax"][key][i][1] is None:
                                constraints["coef_channel_minmax"][key][i] = \
                                    [constraints["coef_channel_minmax"][key][i][0], np.inf]

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

    def __init__(self, dim_volume, affine, constraints, orders, manufacturer="", shim_cs=None,
                 isocenter=np.array([0, 0, 0])):
        """
        Args:
            dim_volume (tuple): x, y and z dimensions.
            affine (np.ndarray): 4x4 array containing the qform affine transformation for the coil profiles
            constraints (dict): dict containing the constraints for the coil profiles. Required keys:

                * name (str): Name of the coil.
                * coef_sum_max (float): Contains the maximum value for the sum of the coefficients. None is used to
                  specify no bounds
                * coef_channel_max (list): List of ``(min, max)`` pairs for each coil channels. (None, None) is
                  used to specify no bounds.

            orders (tuple): Degrees of the desired terms in the series expansion, specified as a vector of non-negative
                            integers (``(0:1:n)`` yields harmonics up to n-th order)
            manufacturer (str): Manufacturer of the scanner. "SIEMENS", "GE" or "PHILIPS".
            shim_cs (str): Coordinate system of the shims. Letter 1 'R' or 'L', letter 2 'A' or 'P', letter 3 'S' or
                           'I'. Only relevant if the manufacturer is unknown. Default: 'RAS'.
            isocenter (np.ndarray): Position of the shim table in the scanner. Default: [0, 0, 0]
        """
        self.orders = orders

        manufacturer = manufacturer.upper()
        if manufacturer in SHIM_CS:
            self.coord_system = SHIM_CS[manufacturer]
        else:
            logger.warning(f"Unknown manufacturer {manufacturer}")
            if shim_cs is None:
                self.coord_system = 'RAS'
            else:
                self.coord_system = shim_cs

        self.affine = affine
        self.isocenter = isocenter

        # Create the spherical harmonics with the correct order, dim and affine
        sph_coil_profile = self._create_coil_profile(dim_volume, manufacturer)
        # Restricts the constraints to the specified order
        constraints['coef_channel_minmax'] = restrict_sph_constraints(constraints['coef_channel_minmax'], self.orders)
        super().__init__(sph_coil_profile, affine, constraints)

    def _create_coil_profile(self, dim, manufacturer=None):
        # Create spherical harmonics coil profiles
        # Change the affine offset so that the origin is at isocenter
        affine_origin_iso = copy.deepcopy(self.affine)
        affine_origin_iso[:3, 3] -= self.isocenter
        mesh1, mesh2, mesh3 = generate_meshgrid(dim, affine_origin_iso)
        if manufacturer == 'SIEMENS':
            sph_coil_profile = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(self.orders))
        elif manufacturer == 'GE':
            sph_coil_profile = ge_basis(mesh1, mesh2, mesh3, orders=tuple(self.orders))
        elif manufacturer == 'PHILIPS':
            sph_coil_profile = philips_basis(mesh1, mesh2, mesh3, orders=tuple(self.orders))
        else:
            logger.warning(f"{manufacturer} manufacturer not implemented. Outputting in Hz, uT/m, uT/m^2, uT/m^3 for "
                           "order 0, 1, 2 and 3 respectively")
            sph_coil_profile = sh_basis(mesh1, mesh2, mesh3, orders=tuple(self.orders), shim_cs=self.coord_system)

        return sph_coil_profile

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)


def get_scanner_constraints(manufacturers_model_name, orders, manufacturer):
    """ Returns the scanner spherical harmonics constraints depending on the manufacturer's model name and required
        order

    Args:
        manufacturers_model_name (str): Name of the scanner
        orders (list): List of all orders of the shim system to be used
        manufacturer (str): Manufacturer of the scanner

    Returns:
        dict: The constraints including the scanner name, bounds and the maximum sum of currents.
    """
    constraints = {
        "coef_channel_minmax": {"0": [], "1": [], "2": [], "3": []},
        "coef_sum_max": None
    }

    # If the manufacturer and scanner is implemented
    if (manufacturer in SCANNER_CONSTRAINTS.keys() and
            manufacturers_model_name in SCANNER_CONSTRAINTS[manufacturer].keys()):
        constraints["name"] = manufacturers_model_name
        for order in orders:
            constrs = SCANNER_CONSTRAINTS[manufacturer][manufacturers_model_name][str(order)]
            if constrs:
                constraints["coef_channel_minmax"][str(order)] = constrs
            else:
                n_channels = channels_per_order(order, manufacturer)
                constraints["coef_channel_minmax"][str(order)] = [[None, None] for _ in range(n_channels)]
                logger.warning(
                    f"Order {order} not available on the {manufacturers_model_name}, unconstrained optimization for "
                    f"this order.")

    else:
        logger.warning(f"Scanner: {manufacturers_model_name} not implemented, constraints might not be respected.")
        constraints["name"] = "Unknown"

        # Fill with Nones
        for order in orders:
            n_channels = channels_per_order(order, manufacturer)
            constraints["coef_channel_minmax"][str(order)] = [[None, None] for _ in range(n_channels)]

    return constraints


def restrict_sph_constraints(bounds: dict, orders):
    """ Select bounds according to the order specified

    Args:
        bounds (dict): Dictionary containing the min and max currents for multiple spherical harmonics orders
        orders (list): List of all spherical harmonics orders to be used

    Returns:
        dict: Dictionary with the bounds of all specified orders
    """
    minmax_out = {}
    for order in orders:
        if f"{order}" in bounds:
            minmax_out[f"{order}"] = bounds[f"{order}"]

    if minmax_out == {}:
        raise NotImplementedError(f"Order must be between 0 and 3")

    return minmax_out
