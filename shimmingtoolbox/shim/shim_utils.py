# -*- coding: utf-8 -*-
"""
This file includes utility functions useful for the shimming module
"""

import nibabel as nib
import json
import numpy as np
import logging

from shimmingtoolbox.coils.coil import SCANNER_CONSTRAINTS, SCANNER_CONSTRAINTS_DAC
from shimmingtoolbox.coils.coordinates import phys_to_vox_coefs, get_main_orientation
from shimmingtoolbox.coils.spher_harm_basis import get_flip_matrix, SHIM_CS, channels_per_order

logger = logging.getLogger(__name__)


def get_phase_encode_direction_sign(fname_nii):
    """ Returns the phase encode direction sign

    Args:
        fname_nii (str): Filename to a NIfTI file with its corresponding json file.

    Returns:
        bool: Returns whether the encoding direction is positive (True) or negative (False)
    """

    # Load nibabel
    nii = nib.load(fname_nii)
    dim_info = nii.header.get_dim_info()

    # Load json
    fname_json = fname_nii.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # json_data['PhaseEncodingDirection'] contains i, j or k then a '-' if the direction is reversed
    phase_en_dir = json_data['PhaseEncodingDirection']

    # Check that dim_info is consistent with PhaseEncodingDirection tag i --> 0, j --> 1, k --> 2
    if (phase_en_dir[0] == 'i' and dim_info[1] != 0) \
            or (phase_en_dir[0] == 'j' and dim_info[1] != 1) \
            or (phase_en_dir[0] == 'k' and dim_info[1] != 2):
        raise RuntimeError("Inconsistency between dim_info of fieldmap and PhaseEncodeDirection tag in the json")

    # Find if the phase encode direction is negative or positive
    if len(phase_en_dir) == 2 and phase_en_dir[1] == '-':
        en_is_positive = False
    elif len(phase_en_dir) == 1:
        en_is_positive = True
    else:
        raise ValueError(f"Unexpected value for PhaseEncodingDirection: {phase_en_dir}")

    return en_is_positive


def phys_to_gradient_cs(coefs_x, coefs_y, coefs_z, fname_anat):
    """ Converts physical coefficients (x, y, z from RAS Coordinate System) to Siemens Gradient Coordinate System

    Args:
        coefs_x (numpy.ndarray): Array containing x coefficients in the physical coordinate system RAS
        coefs_y (numpy.ndarray): Array containing y coefficients in the physical coordinate system RAS
        coefs_z (numpy.ndarray): Array containing z coefficients in the physical coordinate system RAS
        fname_anat (str): Filename of the NIfTI file to convert the data to that Gradient CS

    Returns:
        (tuple): tuple containing:
            * numpy.ndarray: Array containing the data in the gradient CS (frequency/readout)
            * numpy.ndarray: Array containing the data in the gradient CS (phase)
            * numpy.ndarray: Array containing the data in the gradient CS (slice)

    """
    # Load anat
    nii_anat = nib.load(fname_anat)

    # Convert from patient coordinates to image coordinates
    scanner_coil_coef_vox = phys_to_vox_coefs(coefs_x, coefs_y, coefs_z, nii_anat.affine)
    # scanner_coil_coef_vox[0]  # NIfTI dim1, etc

    # Convert from image to freq, phase, slice encoding direction
    dim_info = nii_anat.header.get_dim_info()
    coefs_freq, coefs_phase, coefs_slice = [scanner_coil_coef_vox[dim] for dim in dim_info]

    # To output to the gradient coord system, axes need some inversions. The gradient coordinate system is
    # defined by the frequency, phase and slice encode directions.
    # TODO: More tests, validated for TRA, SAG, COR, no-flip/flipped PE, no rotation

    # Load anat json
    fname_anat_json = fname_anat.rsplit('.nii', 1)[0] + '.json'
    with open(fname_anat_json) as json_file:
        json_anat_data = json.load(json_file)

    if 'ImageOrientationText' in json_anat_data:
        # Tag in private dicom header (0051,100E) indicates the slice orientation, if it exists, it will appear
        # in the json under 'ImageOrientationText' tag
        orientation_text = json_anat_data['ImageOrientationText']
        orientation = orientation_text[:3].upper()
    else:
        # Find orientation with the ImageOrientationPatientDICOM tag, this is less reliable since it can fail
        # if there are 2 highest cosines. It will raise an exception if there is a problem
        orientation = get_main_orientation(json_anat_data['ImageOrientationPatientDICOM'])

    if orientation == 'SAG':
        coefs_slice = -coefs_slice
    elif orientation == 'COR':
        coefs_freq = -coefs_freq
    else:
        # TRA
        pass

    phase_encode_is_positive = get_phase_encode_direction_sign(fname_anat)
    if not phase_encode_is_positive:
        coefs_freq = -coefs_freq
        coefs_phase = -coefs_phase

    return coefs_freq, coefs_phase, coefs_slice


def calculate_metric_within_mask(array, mask, metric='mean', axis=None):
    """ Calculate a metric within a ROI defined by a mask

    Args:
        array (np.ndarray): 3d array
        mask (np.ndarray): 3d array with the same shape as array
        metric (string): Metric to calculate, supported: std, mean, mae, mse, rmse
        axis (int): Axis to perform the metric

    Returns:
        np.ndarray: Array containing the output metrics, if axis is None, the output is a single value
    """
    ma_array = np.ma.array(array, mask=mask == False)
    ma_array = np.ma.array(ma_array, mask=np.isnan(ma_array))
    if metric == 'mean':
        output = np.ma.mean(ma_array, axis=axis)
    elif metric == 'std':
        output = np.ma.std(ma_array, axis=axis)
    elif metric == 'mae':
        output = np.ma.mean(np.ma.abs(ma_array), axis=axis)
    elif metric == 'mse':
        output = np.ma.mean(np.ma.power(ma_array, 2), axis=axis)
    elif metric == 'rmse':
        output = np.ma.sqrt(np.ma.mean(np.ma.power(ma_array, 2), axis=axis))
    else:
        raise NotImplementedError("Metric not implemented")

    # Return nan if the output is masked, this avoids warnings for implicit conversions that could happen later
    if output is np.ma.masked:
        return output.filled(np.nan)

    # If it is a masked array, fill the masked values with nans
    if isinstance(output, np.ma.core.MaskedArray):
        return output.filled(np.nan)

    return output


def phys_to_shim_cs(coefs, manufacturer, orders):
    """Convert a list of coefficients from RAS to the Shim Coordinate System

    Args:
        coefs (np.ndarray): Coefficients in the physical RAS coordinate system of the manufacturer. The first
                            dimension represents the different channels. (indexes 0, 1, 2 --> x, y, z...). If there are
                            more coefficients, they are of higher order and must correspond to the implementation of the
                            manufacturer. i.e. Siemens: *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*
        manufacturer (str): Name of the manufacturer
        orders (tuple): Tuple containing the spherical harmonic orders

    Returns:
        np.ndarray: Coefficients in the shim coordinate system of the manufacturer
    """
    manufacturer = manufacturer.upper()

    if manufacturer.upper() in SHIM_CS:
        flip_mat = get_flip_matrix(SHIM_CS[manufacturer], manufacturer=manufacturer, orders=orders)
        if len(flip_mat) != len(coefs):
            logger.warning("Could not convert between shim and physical coordinate system")
        else:
            coefs = flip_mat * coefs

    else:
        logger.warning(f"Manufacturer: {manufacturer} not implemented for the Shim CS. Coefficients might be wrong.")

    return coefs


def shim_to_phys_cs(coefs, manufacturer, orders):
    """ Convert coefficients from the shim coordinate system to the physical RAS coordinate system

    Args:
        coefs (np.ndarray): 1D list of coefficients in the Shim Coordinate System of the manufacturer. The first
                            dimension represents the different channels. Indexes 0, 1, 2 --> x, y, z... If there are
                            more coefficients, they are of higher order and must correspond to the implementation of the
                            manufacturer. Siemens: *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*
        manufacturer (str): Name of the manufacturer
        orders (tuple): Tuple containing the spherical harmonic orders

    Returns:
        np.ndarray: Coefficients in the physical RAS coordinate system

    """

    # It's sign flips so the same function can be used for shimCS <--> phys RAS
    coefs = phys_to_shim_cs(coefs, manufacturer, orders)

    return coefs


def get_scanner_shim_settings(bids_json_dict, orders):
    """ Get the scanner's shim settings using the BIDS tag ShimSetting and ImagingFrequency and returns it in a
        dictionary. 'orders' is used to check if the different orders are available in the metadata.

    Args:
        bids_json_dict (dict): Bids sidecar as a dictionary
        orders (tuple): Tuple containing the spherical harmonic orders

    Returns:
        dict: Dictionary containing the following keys: '0', '1' '2', '3'. The different orders are
              lists unless the different values could not be populated.
    """

    scanner_shim = {
        '0': None,
        '1': None,
        '2': None,
        '3': None,
        'has_valid_settings': False
    }
    # get_imaging_frequency
    if bids_json_dict.get('ImagingFrequency'):
        scanner_shim['0'] = [int(bids_json_dict.get('ImagingFrequency') * 1e6)]

    # get_shim_orders
    if bids_json_dict.get('ShimSetting'):
        n_shim_values = len(bids_json_dict.get('ShimSetting'))
        if n_shim_values == 3:
            scanner_shim['1'] = bids_json_dict.get('ShimSetting')
            scanner_shim['has_valid_settings'] = True
        elif n_shim_values == 8:
            pass
            scanner_shim['2'] = bids_json_dict.get('ShimSetting')[3:]
            scanner_shim['1'] = bids_json_dict.get('ShimSetting')[:3]
            scanner_shim['has_valid_settings'] = True
        else:
            logger.warning(f"ShimSetting tag has an unsupported number of values: {n_shim_values}")
    else:
        logger.warning("ShimSetting tag is not available")

    # Check if the orders to shim are available in the metadata
    for order in orders:
        if scanner_shim.get(str(order)) is None:
            logger.warning(f"Order {order} shim settings not available in the JSON metadata, constraints might not be "
                           f"respected.")

    return scanner_shim


def dac_to_shim_units(manufacturer, manufacturers_model_name, shim_settings):
    """ Converts the ShimSettings tag from the json BIDS sidecar to the ui units.
        (i.e. For the Prisma fit DAC --> uT/m, uT/m^2 (1st order, 2nd order))

    Args:
        manufacturer (str): Manufacturer of the scanner. "SIEMENS", "GE" or "PHILIPS".
        manufacturers_model_name (str): Name of the model of the scanner. Found in the json BIDS sidecar under
                                        ManufacturersModelName'. Supported names: 'Prisma_fit'.
        shim_settings (dict): Dictionary with keys: 'order1', 'order2', 'has_valid_settings'. 'order1' is a list of 3
                       coefficients for the first order. Found in the json BIDS sidecar under 'ShimSetting'. 'order2' is
                       a list of 5 coefficients. 'has_valid_settings' is a boolean.

    Returns:
        dict: Same dictionary as the shim_settings input with coefficients of the first, second and third order
              converted according to the appropriate manufacturer model.
    """
    scanner_shim_mp = shim_settings

    # Check if the manufacturer is implemented
    if manufacturer not in SCANNER_CONSTRAINTS_DAC.keys():
        logger.warning(f"{manufacturer} not implemented or does not include enough metadata information")

    # Check if the manufacturer model is implemented
    elif manufacturers_model_name in SCANNER_CONSTRAINTS_DAC[manufacturer].keys():
        scanner_constraints_dac = SCANNER_CONSTRAINTS_DAC[manufacturer][manufacturers_model_name]
        scanner_constraints = SCANNER_CONSTRAINTS[manufacturer][manufacturers_model_name]

        # Do all the orders except f0
        for order in ['1', '2', '3']:
            # Make sure the order is available in the metadata
            if shim_settings.get(order):
                # Check if the order is implemented
                if not scanner_constraints_dac.get(order):
                    logger.warning(f"Order {order} conversion of {manufacturers_model_name} not implemented.")
                    continue

                # Convert to ui units
                coefs_dac = shim_settings[order]
                max_coef_ui = np.array([cst[1] for cst in scanner_constraints[order]])
                coefs_ui = (np.array(coefs_dac) * max_coef_ui / np.array(scanner_constraints_dac[order]))
                tolerance = 0.001 * max_coef_ui
                if np.any(np.abs(coefs_ui) > (max_coef_ui + tolerance)):
                    scanner_shim_mp['has_valid_settings'] = False
                    raise ValueError("Current shim settings exceed known system limits.")
                else:
                    scanner_shim_mp[order] = coefs_ui

    else:
        logger.warning(f"Manufacturer model {manufacturers_model_name} not implemented,"
                       f"could not convert shim settings")

    return scanner_shim_mp


class ScannerShimSettings:
    def __init__(self, bids_json_dict, orders=None):

        shim_settings_dac = get_scanner_shim_settings(bids_json_dict, orders)
        manufacturer_model_name = bids_json_dict.get('ManufacturersModelName')
        manufacturer = bids_json_dict.get('Manufacturer')
        self.shim_settings = dac_to_shim_units(manufacturer, manufacturer_model_name, shim_settings_dac)

    def concatenate_shim_settings(self, orders=[2]):
        coefs = []
        if not self.shim_settings['has_valid_settings']:
            logger.warning("Invalid Shim Settings")
            return coefs

        if any(order >= 0 for order in orders):
            for order in sorted(orders):
                if self.shim_settings.get(str(order)) is not None:
                    # Concatenate 2 lists
                    coefs.extend(self.shim_settings.get(str(order)))
                else:
                    n_coefs = channels_per_order(order)
                    coefs.extend([0] * n_coefs)

        return coefs
