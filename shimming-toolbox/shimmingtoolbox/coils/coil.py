#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
from typing import Tuple

from shimmingtoolbox.coils.spher_harm_basis import (sh_basis, siemens_basis, ge_basis, philips_basis, SHIM_CS,
                                                    channels_per_order)
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox import __config_scanner_constraints__

logger = logging.getLogger(__name__)

required_constraints = [
    "name",
    "coef_channel_minmax",
    "coef_sum_max"
]

SIEMENS_PRISMA_CONSTRAINTS_167006 = {
            "0": [[123100100, 123265000]],
            "1": [[-2300, 2300], [-2300, 2300], [-2300, 2300]],
            "2": [[-4959.01, 4959.01], [-3551.29, 3551.29], [-3503.299, 3503.299], [-3551.29, 3551.29],
                  [-3487.302, 3487.302]],
            "3": []
        }

SCANNER_CONSTRAINTS = {
    "Siemens": {
        "MAGNETOM_Prisma_Fit_167006": SIEMENS_PRISMA_CONSTRAINTS_167006,
        "Prisma_fit_167006": SIEMENS_PRISMA_CONSTRAINTS_167006,
        "Investigational_Device_7T_79017": {
            "0": [[296490000, 297490000]],
            "1": [[-2999.899, 2999.399], [-2999.886, 2999.886], [-2999.910, 2999.910]],
            "2": [[-7486.502, 7486.502], [-3743.251, 3743.251], [-3695.261, 3695.261], [-3695.261, 3695.261], [-3647.270, 3647.270]],
            "3": []
        },
        "Investigational_Device_7T_18923": {
            "0": [[296490000, 297490000]],
            "1": [[-4999.976, 4999.976], [-4999.980, 4999.980], [-4999.957, 4999.957]],
            "2": [[-6163.2, 6163.2], [-2592.0, 2592.0], [-2592.0, 2592.0], [-2476.8, 2476.8], [-2476.8, 2476.8]],
            "3": []
        },
        "Terra_00000": {
            "0": [[296760000, 297250000]],
            "1": [[-3000, 3000], [-3000, 3000], [-3000, 3000]],
            "2": [[-9360.0, 9360.0], [-4680.0, 4680.0], [-4620.0, 4620.0], [-4620.0, 4620.0], [-4560.0, 4560.0]],
            "3": [[-15232.0, 15232.0], [-14016.0, 14016.0], [-14016.0, 14016.0], [-14016.0, 14016.0]],
        }
    },
    "GE": {

    },
    "Philips": {
        "Ingenia_Elition_X_45590": {
            "0": [[-np.inf, np.inf]],
            "1": [[-1.000000, 1.000000],
                  [-1.000000, 1.000000],
                  [-1.000000, 1.000000]],
            "2": [[-2.132910, 2.132910],
                  [-5.422375, 5.422375],
                  [-5.280444, 5.280444],
                  [-2.216936, 2.216936],
                  [-2.238412, 2.238412]],
            "3": []
        }
    }
}

SIEMENS_PRISMA_DAC_CONSTRAINTS_167006 = {
    "1": [14436.0, 14265.0, 14045.0],
    "2": [9998.0] * 5,
    "3": []
}

# One can use the Siemens commandline AdjValidate tool to get all the values below
SCANNER_CONSTRAINTS_DAC = {
    "Siemens": {
        "MAGNETOM_Prisma_Fit_167006": SIEMENS_PRISMA_DAC_CONSTRAINTS_167006,
        "Prisma_fit_167006": SIEMENS_PRISMA_DAC_CONSTRAINTS_167006,
        "Investigational_Device_7T_79017": {
            "1": [62479.0, 62264.0, 54082.0],
            "2": [18000.0] * 5,
            "3": []
        },
        "Investigational_Device_7T_18923": {
            "1": [62479.0, 62264.0, 54082.0],
            "2": [18000.0] * 5,
            "3": []
        },
        "Terra_00000": {
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
            constraints (dict): dict containing the constraints for the coil profiles.

                * name (str): Name of the coil. (Required)
                * coef_sum_max (float): Contains the maximum value for the sum of the coefficients. None is used to
                  specify no bounds. (Required)
                * coef_channel_minmax (list): List of ``[min, max]`` pairs for each coil channels. (None, None) is
                  used to specify no bounds. (Required)
                * coefs_used (list): List of the coefficients that are currently being used. Defaults to 0 if not
                  set. (Optional)

        Examples:

            ::

                # Example of constraints
                constraints = {
                    "name": "custom",
                    "coef_channel_minmax": {
                        "coil": [[-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5],
                                 [-2.5, 2.5]]
                    },
                    "coef_sum_max": 20,
                    "coefs_used": {"coil": [1, 1, 1, 1, 1, 1, 1, 1]},
                    "Units": "A"
                }
                constraints = {
                    "name": "Prisma_fit",
                    "coef_channel_minmax": {
                        "0": [[123100100, 123265000]],
                        "1": [[-2300, 2300],
                              [-2300, 2300],
                              [-2300, 2300]],
                        "2": [[-4959.01, 4959.01],
                              [-3551.29, 3551.29],
                              [-3503.299, 3503.299],
                              [-3551.29, 3551.29],
                              [-3487.302, 3487.302]]
                        },
                    "coef_sum_max": None
                    "coefs_used": {
                        "0": [1],
                        "1": [1, 1, 1],
                        "2": [1, 1, 1, 1, 1]
                    },
                }
        """

        self.dim = (np.nan,) * 4
        self.profile = profile
        self.required_constraints = required_constraints

        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")
        self.affine = affine

        self.name = None
        self.coef_channel_minmax = None
        self.coef_sum_max = None
        self.coefs_used = None
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
        """
        Loads the constraints as attribute to this class. The constraints are updated according to the 'coefs_used',
        if available.
        """

        # Read in "coefs_used"
        if constraints.get("coefs_used") is not None:
            self.coefs_used = constraints["coefs_used"]

        for key_name in required_constraints:
            if key_name in constraints:
                if key_name == "coef_channel_minmax":
                    self.coef_channel_minmax = constraints["coef_channel_minmax"]
                    # Error checking
                    if sum([len(constraints["coef_channel_minmax"][key]) for key in
                            constraints["coef_channel_minmax"]]) != self.dim[3]:
                        calculated_n_channels = sum([len(constraints['coef_channel_minmax'][key]) for key in
                                                     constraints['coef_channel_minmax']])
                        raise ValueError(f"length of 'coef_channel_max' must be the same as the number of channels: "
                                         f"{self.dim[3]}, currently: {calculated_n_channels}")

                    for key in constraints["coef_channel_minmax"]:
                        if (self.coefs_used is not None and
                                self.coefs_used.get(key) is not None):
                            # Error checking
                            if constraints["coefs_used"].keys() != constraints["coef_channel_minmax"].keys():
                                raise ValueError("The coil constraints 'coef_channel_minmax' do not have the same keys "
                                                 "as 'coefs_used'")
                            if any(len(self.coefs_used[key]) !=
                                   len(constraints["coef_channel_minmax"][key])
                                   for key in constraints["coef_channel_minmax"]
                                   if self.coefs_used[key] is not None):
                                raise ValueError("The coil's bounds is not the same length as the initial bounds")

                        coef_channel_minmax = self.coef_channel_minmax
                        for i in range(len(constraints["coef_channel_minmax"][key])):
                            if constraints["coef_channel_minmax"][key][i] is None:
                                coef_channel_minmax[key][i] = [-np.inf, np.inf]
                            if constraints["coef_channel_minmax"][key][i][0] is None:
                                coef_channel_minmax[key][i] = [-np.inf, coef_channel_minmax[key][i][1]]
                            if constraints["coef_channel_minmax"][key][i][1] is None:
                                coef_channel_minmax[key][i] = [coef_channel_minmax[key][i][0], np.inf]

                            # Log a warning if the current coefficients are outside the constraints
                            if (self.coefs_used is not None and
                                    self.coefs_used.get(key) is not None and
                                    self.coefs_used[key][i] is not None):
                                if self.coefs_used[key][i] < coef_channel_minmax[key][i][0]:
                                    logger.warning(
                                        f"Initial coef is outside the bounds allowed in the constraints: "
                                        f"{coef_channel_minmax[key][i][0]}, "
                                        f"initial: {constraints['coefs_used'][key][i]}")
                                if self.coefs_used[key][i] > coef_channel_minmax[key][i][1]:
                                    logger.warning(
                                        f"Initial coef is outside the bounds allowed in the constraints: "
                                        f"{coef_channel_minmax[key][i][1]}, "
                                        f"initial: {self.coefs_used[key][i]}")

                                # Adapt constraints based on currently used coefficients
                                # new bound = old bound - current value
                                # eg: [-3, 1] = [-2, 2] - 1
                                coef_channel_minmax[key][i][0] -= self.coefs_used[key][i]
                                coef_channel_minmax[key][i][1] -= self.coefs_used[key][i]

                        self.coef_channel_minmax = coef_channel_minmax

                elif key_name == "coef_sum_max":
                    coef_sum_max = constraints["coef_sum_max"]
                    if constraints["coef_sum_max"] is None:
                        coef_sum_max = np.inf
                    self.coef_sum_max = coef_sum_max
                else:
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
            isocenter (np.ndarray): Position of the isocenter in the image. Default: [0, 0, 0]
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
        if 'coef_channel_minmax' in constraints.keys():
            constraints['coef_channel_minmax'] = restrict_to_orders(constraints['coef_channel_minmax'],
                                                                          self.orders)
        if 'coefs_used' in constraints.keys():
            constraints['coefs_used'] = restrict_to_orders(constraints['coefs_used'], self.orders)
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


def get_scanner_constraints(manufacturers_model_name, orders, manufacturer, device_serial_number, shim_settings,
                            external_constraints=None):
    """ Returns the scanner spherical harmonics constraints depending on the manufacturer's model name and required
        order

    Args:
        manufacturers_model_name (str): Name of the scanner
        orders (list): List of all orders of the shim system to be used
        manufacturer (str): Manufacturer of the scanner
        device_serial_number (str): Serial number of the device, used to identify the scanner
        shim_settings (dict): Dictionary containing the shim settings
        external_constraints (dict): External constraints to be used as priority

    Returns:
        dict: The constraints including the scanner name, bounds and the maximum sum of currents.
    """
    constraints = {
        "coef_channel_minmax": {"0": [], "1": [], "2": [], "3": []},
        "coef_sum_max": None,
        "coefs_used": {}
    }

    # Min max constraints
    if external_constraints is not None:
        external_minmax = external_constraints.get('coef_channel_minmax')
    else:
        external_minmax = None

    scanner_id = f"{manufacturers_model_name}_{device_serial_number}"
    # If the manufacturer and scanner is implemented
    if manufacturer in SCANNER_CONSTRAINTS.keys() and scanner_id in SCANNER_CONSTRAINTS[manufacturer].keys():
        constraints["name"] = scanner_id
        for order in orders:
            internal_constrs = copy.deepcopy(SCANNER_CONSTRAINTS[manufacturer][scanner_id][str(order)])
            if external_minmax is not None:
                ext_constraints = external_minmax.get(str(order))
            else:
                ext_constraints = None

            if ((internal_constrs and internal_constrs is not None) and
                    (ext_constraints and ext_constraints is not None)):
                logger.warning(f"Scanner constraints for order {order} is defined in both the constraint file "
                               f"and internally. Choosing the ones from the constraint file.")
                constraints["coef_channel_minmax"][str(order)] = ext_constraints
            elif ((not internal_constrs or internal_constrs is None) and
                  (ext_constraints and ext_constraints is not None)):
                constraints["coef_channel_minmax"][str(order)] = ext_constraints
            elif ((internal_constrs and internal_constrs is not None) and
                  (not ext_constraints or ext_constraints is None)):
                constraints["coef_channel_minmax"][str(order)] = internal_constrs
            else:
                n_channels = channels_per_order(order, manufacturer)
                constraints["coef_channel_minmax"][str(order)] = [[None, None] for _ in range(n_channels)]
                logger.warning(
                    f"Order {order} not available on {scanner_id}, unconstrained optimization for "
                    f"this order. Consider defining the constraints in an external constraint file. See "
                    f"{__config_scanner_constraints__}")

    elif external_minmax is not None:
        if external_constraints.get("name") is not None:
            constraints["name"] = external_constraints["name"]
        else:
            constraints["name"] = "Unknown"

        for order in orders:
            if external_minmax.get(str(order)) is None:
                n_channels = channels_per_order(order, manufacturer)
                constraints["coef_channel_minmax"][str(order)] = [[None, None] for _ in range(n_channels)]
            else:
                constraints["coef_channel_minmax"][str(order)] = external_minmax.get(str(order))

    else:
        logger.warning(f"Scanner: {scanner_id} not implemented, constraints are not known internally. "
                       f"Consider defining the constraints in an external constraint file. See "
                       f"{__config_scanner_constraints__}")
        constraints["name"] = "Unknown"

        # Fill with Nones
        for order in orders:
            n_channels = channels_per_order(order, manufacturer)
            constraints["coef_channel_minmax"][str(order)] = [[None, None] for _ in range(n_channels)]

    # Coefs used constraint
    if external_constraints is not None:
        external_coefs_used = external_constraints.get('coefs_used')
    else:
        external_coefs_used = None

    if external_coefs_used is not None:
        for order in orders:
            if ((shim_settings[str(order)] is not None) and (external_coefs_used.get(str(order)) is not None)):
                logger.warning(f"Scanner Shim Settings for order {order} is defined in both the constraint file "
                               f"and the BIDS JSON sidecar. Choosing the Shim Settings from the constraint file.")
                constraints['coefs_used'][str(order)] = external_coefs_used.get(str(order))
            elif shim_settings[str(order)] is None and external_coefs_used.get(str(order)) is not None:
                constraints['coefs_used'][str(order)] = external_coefs_used.get(str(order))
            elif shim_settings[str(order)] is not None and external_coefs_used.get(str(order)) is None:
                constraints['coefs_used'][str(order)] = shim_settings[str(order)]
            else:
                logger.warning(f"Scanner Shim Settings for order {order} is not defined in a constraint file or "
                               f"in the BIDS JSON sidecar. Consider adding them manually using 'coefs_used'")
                n_channels = channels_per_order(order, manufacturer)
                constraints['coefs_used'][str(order)] = [None for _ in range(n_channels)]
    else:
        constraints['coefs_used'] = restrict_to_orders(shim_settings, orders)

    return constraints


def restrict_to_orders(shim_dict: dict, orders):
    """ Select the keys according to the order specified

    Args:
        shim_dict (dict): Dictionary containing keys with the spherical harmonic orders
        orders (list): List of all spherical harmonics orders to be used

    Returns:
        dict: Dictionary with only the keys specified in orders
    """
    minmax_out = {}
    for order in orders:
        if f"{order}" in shim_dict:
            minmax_out[f"{order}"] = shim_dict[f"{order}"]

    if minmax_out == {}:
        raise NotImplementedError(f"Order must be between 0 and 3")

    return minmax_out
