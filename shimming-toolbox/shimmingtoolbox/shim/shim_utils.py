# -*- coding: utf-8 -*-
"""
This file includes utility functions useful for the shimming module
"""
import copy
import nibabel as nib
import json
import numpy as np
import logging
from nibabel.affines import apply_affine

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


def phys_to_gradient_cs(coefs_x, coefs_y, coefs_z, fname_target):
    """ Converts physical coefficients (x, y, z from RAS Coordinate System) to Siemens Gradient Coordinate System

    Args:
        coefs_x (numpy.ndarray): Array containing x coefficients in the physical coordinate system RAS
        coefs_y (numpy.ndarray): Array containing y coefficients in the physical coordinate system RAS
        coefs_z (numpy.ndarray): Array containing z coefficients in the physical coordinate system RAS
        fname_target (str): Filename of the NIfTI file to convert the data to that Gradient CS

    Returns:
        (tuple): tuple containing:
            * numpy.ndarray: Array containing the data in the gradient CS (frequency/readout)
            * numpy.ndarray: Array containing the data in the gradient CS (phase)
            * numpy.ndarray: Array containing the data in the gradient CS (slice)

    """
    # Load target
    nii_target = nib.load(fname_target)

    # Convert from patient coordinates to image coordinates
    scanner_coil_coef_vox = phys_to_vox_coefs(coefs_x, coefs_y, coefs_z, nii_target.affine)
    # scanner_coil_coef_vox[0]  # NIfTI dim1, etc

    # Convert from image to freq, phase, slice encoding direction
    dim_info = nii_target.header.get_dim_info()
    coefs_freq, coefs_phase, coefs_slice = [scanner_coil_coef_vox[dim] for dim in dim_info]

    # To output to the gradient coord system, axes need some inversions. The gradient coordinate system is
    # defined by the frequency, phase and slice encode directions.
    # TODO: More tests, validated for TRA, SAG, COR, no-flip/flipped PE, no rotation

    # Load target json
    fname_target_json = fname_target.rsplit('.nii', 1)[0] + '.json'
    with open(fname_target_json) as json_file:
        json_target_data = json.load(json_file)

    if 'ImageOrientationText' in json_target_data:
        # Tag in private dicom header (0051,100E) indicates the slice orientation, if it exists, it will appear
        # in the json under 'ImageOrientationText' tag
        orientation_text = json_target_data['ImageOrientationText']
        orientation = orientation_text[:3].upper()
    else:
        # Find orientation with the ImageOrientationPatientDICOM tag, this is less reliable since it can fail
        # if there are 2 highest cosines. It will raise an exception if there is a problem
        orientation = get_main_orientation(json_target_data['ImageOrientationPatientDICOM'])

    if orientation == 'SAG':
        coefs_slice = -coefs_slice
    elif orientation == 'COR':
        coefs_freq = -coefs_freq
    else:
        # TRA
        pass

    phase_encode_is_positive = get_phase_encode_direction_sign(fname_target)
    if not phase_encode_is_positive:
        coefs_freq = -coefs_freq
        coefs_phase = -coefs_phase

    return coefs_freq, coefs_phase, coefs_slice


def calculate_metric_within_mask(array, mask, metric, axis=None):
    """Calculate a weighted metric within a region of interest (ROI) defined by a mask.

    This function computes various metrics (mean, standard deviation, mean absolute error,
    mean squared error, root mean squared error) over a 3D array, considering only the non-zero
    elements within the mask. The mask contains values from 0 to 1, where 0 indicates
    the data is masked. For values between 0 and 1, the data is weighted accordingly.

    Args:
        array (np.ndarray): 3D array of numerical values to compute the metric on.
        mask (np.ndarray): 3D array with the same shape as `array`, with values between 0 and 1
                           that define the region of interest (ROI).
        metric (str): The metric to calculate. Options are:
                      'mean' (average), 'std' (standard deviation),
                      'mae' (mean absolute error), 'mse' (mean squared error),
                      'rmse' (root mean squared error).
        axis (int or None): Axis to compute the metric.

    Returns:
        np.ndarray: Array containing the output metrics, if axis is None, the output is a single value
    """
    ma_array = np.ma.array(array, mask=mask == 0)
    ma_array = np.ma.array(ma_array, mask=np.isnan(ma_array))

    # Prevent division by zero
    if np.ma.sum(mask) == 0:
        return np.nan

    if metric == 'mean':
        output = np.ma.average(ma_array, weights=mask, axis=axis)

    elif metric == 'std':
        mean_weighted = np.ma.average(ma_array, weights=mask, axis=axis)
        variance = np.ma.average(np.ma.power(ma_array - mean_weighted, 2), weights=mask, axis=axis)
        output = np.ma.sqrt(variance)

    elif metric == 'mae':
        abs_diff = np.ma.abs(ma_array)
        output = np.ma.average(abs_diff, weights=mask, axis=axis)

    elif metric == 'mse' :
        squared_diff = np.ma.power(ma_array, 2)
        output = np.ma.average(squared_diff, weights=mask, axis=axis)

    elif metric == 'rmse':
        squared_diff = np.ma.power(ma_array, 2)
        output = np.ma.sqrt(np.ma.average(squared_diff, weights=mask, axis=axis))

    else:
        raise NotImplementedError(f"Metric '{metric}' not implemented. Available metrics: 'mean', 'std', 'mae', 'mse', 'rmse'.")

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
            raise ValueError("Could not convert between shim and physical coordinate system")
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


def dac_to_shim_units(manufacturer, manufacturers_model_name, device_serial_number, shim_settings):
    """ Converts the ShimSettings tag from the json BIDS sidecar to the ui units.
        (i.e. For the Prisma fit DAC --> uT/m, uT/m^2 (1st order, 2nd order))

    Args:
        manufacturer (str): Manufacturer of the scanner. "SIEMENS", "GE" or "PHILIPS".
        manufacturers_model_name (str): Name of the model of the scanner. Found in the json BIDS sidecar under
                                        ManufacturersModelName'. Supported names: 'Prisma_fit'.
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


def convert_to_dac_units(shim_settings_coefs_ui, scanner_constraints, scanner_constraints_dac):
    """ Convert shim settings from ui units to DAC units

    Args:
        shim_settings_coefs_ui (list): List of coefficients in the ui units
        scanner_constraints (list): List containing the constraints of the scanner for a specific order
        scanner_constraints_dac (list): List containing the maximum DAC values for a specific order

    Returns:
        list: List of coefficients in the DAC units
    """
    # Convert to dac units
    max_coefs_ui = np.array([cst[1] for cst in scanner_constraints])
    min_coefs_ui = np.array([cst[0] for cst in scanner_constraints])
    coefs_dac = (np.array(shim_settings_coefs_ui) * 2 * np.array(scanner_constraints_dac) /
                 (max_coefs_ui - min_coefs_ui))
    tolerance = 0.001 * scanner_constraints_dac
    if (np.any(coefs_dac > (scanner_constraints_dac + tolerance)) or
            np.any(coefs_dac < (-scanner_constraints_dac - tolerance))):
        logger.warning("Future shim settings exceed known system limits.")

    return coefs_dac


def extend_slice(nii_array, n_slices=1, axis=2, location=None):
    """
    Adds n_slices on each side of the selected axis. It uses the nearest slice and copies it to fill the values.
    Updates the affine of the matrix to keep the input array in the same location.

    Args:
        nii_array (nib.Nifti1Image): 3d or 4d array to extend the dimensions along an axis.
        n_slices (int): Number of slices to add on each side of the selected axis.
        axis (int): Axis along which to insert the slice(s), Allowed axis: 0, 1, 2.
        location (np.array): Location where the original data is located in the new data.
    Returns:
        nib.Nifti1Image: Array extended with the appropriate affine to conserve where the original pixels were located.

    Examples:
        ::
            print(nii_array.get_fdata().shape)  # (50, 50, 1, 10)
            nii_out = extend_slice(nii_array, n_slices=1, axis=2)
            print(nii_out.get_fdata().shape)  # (50, 50, 3, 10)
    """
    # Locate original data in new data
    orig_data_in_new_data = location

    if nii_array.get_fdata().ndim == 3:
        extended = nii_array.get_fdata()
        extended = extended[..., np.newaxis]
        if location is not None:
            orig_data_in_new_data = orig_data_in_new_data[..., np.newaxis]
    elif nii_array.get_fdata().ndim == 4:
        extended = nii_array.get_fdata()
    else:
        raise ValueError("Unsupported number of dimensions for input array")

    for i_slice in range(n_slices):
        if axis == 0:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros(orig_data_in_new_data.shape[1:]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros(orig_data_in_new_data.shape[1:]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[-1, :, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[0, :, :, :], axis=axis)
        elif axis == 1:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros_like(orig_data_in_new_data[:, 0, :, :]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros_like(orig_data_in_new_data[:, 0, :, :]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[:, -1, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, 0, :, :], axis=axis)
        elif axis == 2:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros_like(orig_data_in_new_data[:, :, 0, :]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros_like(orig_data_in_new_data[:, :, 0, :]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[:, :, -1, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, :, 0, :], axis=axis)
        else:
            raise ValueError("Unsupported value for axis")

    new_affine = update_affine_for_ap_slices(nii_array.affine, n_slices, axis)

    if nii_array.get_fdata().ndim == 3:
        extended = extended[..., 0]

    nii_extended = nib.Nifti1Image(extended, new_affine, header=nii_array.header)

    if location is not None:
        return nii_extended, orig_data_in_new_data

    return nii_extended


def update_affine_for_ap_slices(affine, n_slices=1, axis=2):
    """
    Updates the input affine to reflect an insertion of n_slices on each side of the selected axis

    Args:
        affine (np.ndarray): 4x4 qform affine matrix representing the coordinates
        n_slices (int): Number of pixels to add on each side of the selected axis
        axis (int): Axis along which to insert the slice(s)
    Returns:
        np.ndarray: 4x4 updated affine matrix
    """
    # Define indexes
    index_shifted = [0, 0, 0]
    index_shifted[axis] = n_slices

    # Difference of voxel in world coordinates
    spacing = apply_affine(affine, index_shifted) - apply_affine(affine, [0, 0, 0])

    # Calculate new affine
    new_affine = affine
    new_affine[:3, 3] = affine[:3, 3] - spacing

    return new_affine


class ScannerShimSettings:
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
        coefs = []

        if any(order >= 0 for order in orders):
            for order in sorted(orders):
                if self.shim_settings.get(str(order)) is not None:
                    # Concatenate 2 lists
                    coefs.extend(self.shim_settings.get(str(order)))
                else:
                    n_coefs = channels_per_order(order)
                    coefs.extend([0] * n_coefs)

        return coefs
