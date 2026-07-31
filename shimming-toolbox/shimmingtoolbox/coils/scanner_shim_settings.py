# -*- coding: utf-8 -*-
"""
This file includes functions and classes useful to extract scanner's shim settings
"""
import copy
import logging
import numpy as np

from shimmingtoolbox.coils.coil import SCANNER_CONSTRAINTS, SCANNER_CONSTRAINTS_DAC
from shimmingtoolbox.coils.spher_harm_basis import channels_per_order

logger = logging.getLogger(__name__)


class ScannerShimSettings:
    """ Class to handle the scanner shim settings from a NIfTI fieldmap file.
    """
    def __init__(self, nif_fmap, orders=None):

        shim_settings_dac = nif_fmap.get_scanner_shim_settings(orders=orders)
        manufacturers_model_name = nif_fmap.get_manufacturers_model_name()
        manufacturer = nif_fmap.get_json_info('Manufacturer')
        device_serial_number = nif_fmap.get_json_info('DeviceSerialNumber')

        self.shim_settings = dac_to_shim_units(manufacturer,
                                               manufacturers_model_name,
                                               device_serial_number,
                                               shim_settings_dac)

    def concatenate_shim_settings(self, orders=[2]):
        return concatenate_shim_settings(self.shim_settings, orders=orders)


def concatenate_shim_settings(shim_settings, orders=[2]):
    """

    Args:
        shim_settings (dict): Dictionary of shimSettings.
        orders (list): List of orders to concatenate.

    Returns:
        list: List of coefficients concatenated in the order of the orders. If an order is not available,
              it will be filled with zeros.
    """
    coefs = []

    if any(order >= 0 for order in orders):
        for order in sorted(orders):
            if shim_settings.get(str(order)) is not None:
                # Concatenate 2 lists
                coefs.extend(shim_settings.get(str(order)))
            else:
                n_coefs = channels_per_order(order)
                coefs.extend([0] * n_coefs)

    return coefs


def dac_to_shim_units(manufacturer, manufacturers_model_name, device_serial_number, shim_settings):
    """ Converts the ShimSettings tag from the json BIDS sidecar to the ui units.
        (i.e. For the Prisma fit DAC --> uT/m, uT/m^2 (1st order, 2nd order))

    Args:
        manufacturer (str): Manufacturer of the scanner. "SIEMENS", "GE" or "PHILIPS".
        manufacturers_model_name (str): Name of the model of the scanner. Found in the json BIDS sidecar under
                                        'ManufacturersModelName'. Supported names: 'Prisma_fit'.
        device_serial_number (str): Serial number of the scanner. Found in the json BIDS sidecar under
                                    DeviceSerialNumber.
        shim_settings (dict): Dictionary with keys: '1', '2'. Found in the json BIDS sidecar under 'ShimSetting'. '2' is
                       a list of 5 coefficients.

    Returns:
        dict: Same dictionary as the shim_settings input with coefficients of the first, second and third order
              converted according to the appropriate manufacturer model.
    """
    scanner_shim_mp = copy.deepcopy(shim_settings)

    scanner_id = f"{manufacturers_model_name}_{device_serial_number}"

    # Check if the manufacturer is implemented
    if manufacturer not in SCANNER_CONSTRAINTS_DAC.keys():
        logger.warning(f"{manufacturer} not implemented or does not include enough metadata information")

    # Check if the scanner_id is implemented
    elif scanner_id in SCANNER_CONSTRAINTS_DAC[manufacturer].keys():
        scanner_constraints_dac = SCANNER_CONSTRAINTS_DAC[manufacturer][scanner_id]
        scanner_constraints = SCANNER_CONSTRAINTS[manufacturer][scanner_id]

        # Do all the orders except f0
        for order in ['0', '1', '2', '3']:
            # Make sure the order is available in the metadata
            if shim_settings.get(order) and shim_settings[order] is not None:

                # No conversion necessary for f0
                if order == '0':
                    # F0 is in Hz, no conversion necessary, just check that the current frequency fits within the bounds
                    max_0 = scanner_constraints[order][0][1]
                    min_0 = scanner_constraints[order][0][0]
                    tolerance = 0.001 * (max_0 - min_0)
                    if (shim_settings[order][0] > (max_0 + tolerance)) or (
                            shim_settings[order][0] < (min_0 - tolerance)):
                        raise ValueError(f"Current f0 frequency {shim_settings[order][0]} exceeds known system limits.")
                    continue
                # Check if unit conversion for the order is implemented
                elif not scanner_constraints_dac.get(order):
                    logger.warning(f"Order {order} conversion of {scanner_id} not implemented.")
                    scanner_shim_mp[order] = None
                    continue

                # Convert the shim settings to ui units
                scanner_shim_mp[order] = _convert_to_ui_units(shim_settings[order],
                                                              scanner_constraints[order],
                                                              scanner_constraints_dac[order])

    else:
        logger.debug(f"Manufacturer model {scanner_id} not implemented, "
                     f"could not convert shim settings")

    return scanner_shim_mp


def _convert_to_ui_units(shim_settings_coefs, scanner_constraints, scanner_constraints_dac):
    # Convert to ui units
    coefs_dac = shim_settings_coefs
    max_coefs_ui = np.array([cst[1] for cst in scanner_constraints])
    min_coefs_ui = np.array([cst[0] for cst in scanner_constraints])
    coefs_ui = (np.array(coefs_dac) * (max_coefs_ui - min_coefs_ui) / (2 * np.array(scanner_constraints_dac)))
    tolerance = 0.001 * (max_coefs_ui - min_coefs_ui)
    if np.any(coefs_ui > (max_coefs_ui + tolerance)) or np.any(coefs_ui < (min_coefs_ui - tolerance)):
        raise ValueError("Current shim settings exceed known system limits.")

    return coefs_ui
