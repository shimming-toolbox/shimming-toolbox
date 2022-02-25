# -*- coding: utf-8 -*-
"""
This file includes utility functions useful for the shimming module
"""

import nibabel as nib
import json
import numpy as np
import logging

from shimmingtoolbox.coils.coordinates import phys_to_vox_coefs, get_main_orientation

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
        metric (string): Metric to calculate, supported: std, mean
        axis (int): Axis to perform the metric

    Returns:
        np.ndarray: Array containing the output metrics, if axis is None, the output is a single value
    """
    ma_array = np.ma.array(array, mask=mask == False)

    if metric == 'mean':
        output = np.ma.mean(ma_array, axis=axis)
    elif metric == 'std':
        output = np.ma.std(ma_array, axis=axis)
    else:
        raise NotImplementedError("Metric not implemented")

    return output


def phys_to_shim_cs(coefs, manufacturer):
    """Convert a list of coefficients from RAS to the Shim Coordinate System

    Args:
        coefs (np.ndarray): 1d list of coefficients in the physical RAS coordinate system of the manufacturer. The first
                            dimension represents the different channels. (indexes 0, 1, 2 --> x, y, z...). If there are
                            more coefficients, they are of higher order and must correspond to the implementation of the
                            manufacturer. i.e. Siemens: *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*
        manufacturer (str): Name of the manufacturer

    Returns:
        np.ndarray: Coefficients in the shim coordinate system of the manufacturer
    """

    if manufacturer == 'Siemens':
        # X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
        # 0, 1, 2, 3,  4,  5,  6,     7

        # Order 1
        if len(coefs) >= 3:
            # Change from RAS to LAI (ShimCS)
            coefs[0] = -coefs[0]  # X
            coefs[2] = -coefs[2]  # Z

        # Order 2
        if len(coefs) >= 8:
            # Invert X and Z --> invert ZY and XY (Z2, XZ and X2-Y2 are double inverted)
            coefs[5] = -coefs[5]  # [ZY]
            coefs[7] = -coefs[7]  # [XY]

    else:
        logger.warning(f"Manufacturer: {manufacturer} not implemented for the Shim CS. Coefficients might be wrong.")

    return coefs


def shim_to_phys_cs(coefs, manufacturer):
    """ Convert coefficients from the shim coordinate system to the physical RAS coordinate system

    Args:
        coefs (np.ndarray): 1D list of coefficients in the Shim Coordinate System of the manufacturer. The first
                            dimension represents the different channels. Indexes 0, 1, 2 --> x, y, z... If there are
                            more coefficients, they are of higher order and must correspond to the implementation of the
                            manufacturer. Siemens: *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*
        manufacturer (str): Name of the manufacturer

    Returns:
        np.ndarray: Coefficients in the physical RAS coordinate system
    """
    # It's sign flips so the same function can be used for shimCS <--> phys RAS
    coefs = phys_to_shim_cs(coefs, manufacturer)

    return coefs
