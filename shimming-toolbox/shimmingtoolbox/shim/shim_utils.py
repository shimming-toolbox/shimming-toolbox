# -*- coding: utf-8 -*-
"""
This file includes utility functions useful for the shimming module
"""
import nibabel as nib
import json
import numpy as np
import logging
from nibabel.affines import apply_affine

from shimmingtoolbox.coils.coordinates import phys_to_vox_coefs, vox_to_phys_coefs, get_main_orientation

logger = logging.getLogger(__name__)

MANUFACTURERS = ('SIEMENS', 'GE', 'PHILIPS')
SHIM_CS = {'SIEMENS': 'LAI',
           'GE': 'LPI',
           'PHILIPS': 'RPI'}

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
    """ Converts physical coefficients (x, y, z from RAS World Coordinate System) to Siemens Gradient Coordinate System (RO, PE, SL)

    Args:
        coefs_x (numpy.ndarray): Array containing x coefficients in the World coordinate system RAS
        coefs_y (numpy.ndarray): Array containing y coefficients in the World coordinate system RAS
        coefs_z (numpy.ndarray): Array containing z coefficients in the World coordinate system RAS
        fname_target (str): Filename of the NIfTI file to convert the data to that Gradient CS

    Returns:
        (tuple): tuple containing:
            * numpy.ndarray: Array containing the data in the gradient CS (frequency/readout)
            * numpy.ndarray: Array containing the data in the gradient CS (phase)
            * numpy.ndarray: Array containing the data in the gradient CS (slice)

    Notes:

        This function transforms the affine to create an image coordinate system that is in line with the
        Gradient coordinate system on Siemens.

        The previous way to do this was flawed if the affine was different than what dcm2niix would output for the
        different orientations (TRA: LAS, SAG: PSR, COR: LSP).
        ::

            scanner_coil_coef_vox = phys_to_vox_coefs(coefs_x, coefs_y, coefs_z, nii_target.affine)
            # TRA: RAS -> LAS  # Assumption that voxel coordinate system is LAS
            # SAG: RAS -> PSR
            # COR: RAS -> LSP

            # Convert from image to frequency, phase, slice encoding direction
            dim_info = nii_target.header.get_dim_info()
            coefs_freq, coefs_phase, coefs_slice = [scanner_coil_coef_vox[dim] for dim in dim_info]
            # TRA: LAS -> LAS, # SAG: PSR -> SPR, # COR: LSP -> SLP

            if orientation == 'SAG':
                coefs_slice = -coefs_slice
            elif orientation == 'COR':
                coefs_freq = -coefs_freq
            # TRA: LAS -> LAS, # SAG: SPR -> SPL, # COR: SLP -> ILP

            if not phase_encode_is_positive:
                coefs_freq = -coefs_freq
                coefs_phase = -coefs_phase
            # PE +: # TRA: LAS, # SAG: SPL, # COR: ILP
            # PE -: # TRA: LAS -> RPS, # SAG: SPL -> IAL, # COR: ILP -> SRP
    """

    fname_target_json = fname_target.rsplit('.nii', 1)[0] + '.json'
    with open(fname_target_json) as json_file:
        json_target_data = json.load(json_file)

    if 'Manufacturer' not in json_target_data:
        raise ValueError("Manufacturer not found in the json file.")
    manufacturer = json_target_data['Manufacturer']

    if 'PatientPosition' not in json_target_data:
        raise ValueError("PatientPosition not found in json file.")
    patient_position = json_target_data['PatientPosition']

    if 'ImageOrientationText' in json_target_data:
        # Tag in private dicom header (0051,100E) indicates the slice orientation, if it exists, it will appear
        # in the json under 'ImageOrientationText' tag
        orientation_text = json_target_data['ImageOrientationText']
        orientation = orientation_text[:3].upper()
    else:
        # Find orientation with the ImageOrientationPatientDICOM tag, this is less reliable since it can fail
        # if there are 2 highest cosines. It will raise an exception if there is a problem
        orientation = get_main_orientation(json_target_data['ImageOrientationPatientDICOM'])

    phase_encode_is_positive = get_phase_encode_direction_sign(fname_target)

    nii_target = nib.load(fname_target)
    new_affine = convert_affine_to_gradient_cs(manufacturer,
                                               patient_position,
                                               orientation,
                                               phase_encode_is_positive,
                                               nii_target.affine,
                                               nii_target.shape)

    coefs_freq, coefs_phase, coefs_slice = phys_to_vox_coefs(coefs_x, coefs_y, coefs_z, new_affine)

    return coefs_freq, coefs_phase, coefs_slice


def gradient_to_phys_cs(coefs_freq, coefs_phase, coefs_slice, fname_target):
    """ Converts Siemens Gradient Coordinate System (RO, PE, SL) to physical coordinates (x, y, z from RAS World Coordinate System)
    See phys_to_gradient_cs for more details. This is the inverse of that function

    Args:
        coefs_freq (numpy.ndarray): Array containing RO coefficients in the Siemens Gradient Coordinate System
        coefs_phase (numpy.ndarray): Array containing PE coefficients in the Siemens Gradient Coordinate System
        coefs_slice (numpy.ndarray): Array containing SL coefficients in the Siemens Gradient Coordinate System
        fname_target (str): Filename of the NIfTI file to convert the data to RAS World Coordinate System

    Returns:
        (tuple): tuple containing:
            * numpy.ndarray: Array containing the data in the x
            * numpy.ndarray: Array containing the data in the y
            * numpy.ndarray: Array containing the data in the z
    """
    fname_target_json = fname_target.rsplit('.nii', 1)[0] + '.json'
    with open(fname_target_json) as json_file:
        json_target_data = json.load(json_file)

    if 'Manufacturer' not in json_target_data:
        raise ValueError("Manufacturer not found in the json file.")
    manufacturer = json_target_data['Manufacturer']

    if 'PatientPosition' not in json_target_data:
        raise ValueError("PatientPosition not found in json file.")
    patient_position = json_target_data['PatientPosition']

    if 'ImageOrientationText' in json_target_data:
        # Tag in private dicom header (0051,100E) indicates the slice orientation, if it exists, it will appear
        # in the json under 'ImageOrientationText' tag
        orientation_text = json_target_data['ImageOrientationText']
        orientation = orientation_text[:3].upper()
    else:
        # Find orientation with the ImageOrientationPatientDICOM tag, this is less reliable since it can fail
        # if there are 2 highest cosines. It will raise an exception if there is a problem
        orientation = get_main_orientation(json_target_data['ImageOrientationPatientDICOM'])

    phase_encode_is_positive = get_phase_encode_direction_sign(fname_target)

    nii_target = nib.load(fname_target)
    new_affine = convert_affine_to_gradient_cs(manufacturer,
                                               patient_position,
                                               orientation,
                                               phase_encode_is_positive,
                                               nii_target.affine,
                                               nii_target.shape)

    coefs_x, coefs_y, coefs_z = vox_to_phys_coefs(coefs_freq, coefs_phase, coefs_slice, new_affine)
    return coefs_x, coefs_y, coefs_z


def convert_affine_to_gradient_cs(manufacturer: str, patient_position: str, orientation: str,
                                  phase_encode_is_positive: bool, affine, shape):
    """ Output the affine matrix to transform from the gradient CS to RAS World Coordinate System.

    Args:
        manufacturer (str): Manufacturer
        patient_position (str): Only HFS is supported
        orientation (str): TRA or SAG or COR
        phase_encode_is_positive (bool): Whether the phase encoding is positive or negative
        affine (np.ndarray): Affine matrix
        shape (np.ndarray): Dimensions of the image

    Returns:
        np.ndarray: New affine matrix to transform from the gradient CS to RAS World Coordinate System
    """
    if manufacturer != 'Siemens':
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported. Only Siemens supported.")
    if patient_position != 'HFS':
        raise NotImplementedError(f"PatientPosition {patient_position} is not supported. Only HFS is implemented")

    if orientation == 'TRA':
        if phase_encode_is_positive:
            target_ornt = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
        else:
            target_ornt = nib.orientations.axcodes2ornt(('R', 'P', 'S'))
    elif orientation == 'SAG':
        if phase_encode_is_positive:
            target_ornt = nib.orientations.axcodes2ornt(('S', 'P', 'L'))
        else:
            target_ornt = nib.orientations.axcodes2ornt(('I', 'A', 'L'))
    elif orientation == 'COR':
        if phase_encode_is_positive:
            target_ornt = nib.orientations.axcodes2ornt(('I', 'L', 'P'))
        else:
            target_ornt = nib.orientations.axcodes2ornt(('S', 'R', 'P'))
    else:
        raise RuntimeError(f"Unexpected value for orientation: {orientation}")

    start_ornt = nib.orientations.axcodes2ornt(nib.aff2axcodes(affine))
    transform_ornt = nib.orientations.ornt_transform(start_ornt, target_ornt)
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform_ornt, shape)
    return new_affine


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


def get_flip_matrix(shim_cs='RAS', manufacturer=None, orders=None):
    f"""
    Return a matrix to flip the spherical harmonics basis set from RAS to the desired coordinate system.

    Args:
        shim_cs (str): Coordinate system of the shim basis set. Default is RAS.
        orders (list): List of orders of the spherical harmonics. Default to None (all orders)
        manufacturer (str): Manufacturer of the scanner. The flipping matrix order is different for each manufacturer.
                            If None is selected, it will output according to
                            ``shimmingtoolbox.coils.spherical_harmonics``. Possible values: {MANUFACTURERS}.

    Returns:
        numpy.ndarray: Matrix (len: 8) to flip the spherical harmonics basis set from ras to the desired coordinate
                       system. Output is a 1D vector of ``flip_matrix`` for the following:
                       Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, YZ2, Z3, XZ^2, Z(X2 - Y2), X(X2 - Y2).
                       If xyz is True, output X, Y, Z only in this order.
    """
    if orders is None:
        orders = [1, 2, 3]

    xyz_cs = [1, 1, 1]

    shim_cs = shim_cs.upper()
    if (len(shim_cs) != 3) or \
            (shim_cs[0] not in ['R', 'L']) or (shim_cs[1] not in ['A', 'P']) or (shim_cs[2] not in ['S', 'I']):
        raise ValueError(f"Unknown coordinate system: {shim_cs}")

    if shim_cs[0] == 'L':
        xyz_cs[0] = -1
    if shim_cs[1] == 'P':
        xyz_cs[1] = -1
    if shim_cs[2] == 'I':
        xyz_cs[2] = -1

    # Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, Z2Y, Z3, Z2X, Z(X2 - Y2), X(X2 - Y2)
    out_dict = {}
    for order in orders:
        if order == 1:
            out_dict[1] = np.array([xyz_cs[1], xyz_cs[2], xyz_cs[0]])
        if order == 2:
            out_dict[2] = np.array([xyz_cs[0] * xyz_cs[1], xyz_cs[2] * xyz_cs[1], 1, xyz_cs[2] * xyz_cs[0], 1])
        if order == 3:
            out_dict[3] = np.array([xyz_cs[1], xyz_cs[0] * xyz_cs[1] * xyz_cs[2], xyz_cs[1], xyz_cs[2], xyz_cs[0],
                                    xyz_cs[2], xyz_cs[0]])

    if manufacturer is not None:
        manufacturer = manufacturer.upper()

    out_dict = reorder_to_manufacturer(out_dict, manufacturer)

    out_list = []
    for i_order in sorted(orders):
        out_list += out_dict[i_order].tolist()

    # None: Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, Z2Y, Z3, Z2X, Z(X2 - Y2), X(X2 - Y2)
    # GE: x, y, z, xy, zy, zx, X2 - Y2, z2, 3rd order not implemented
    # Siemens: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, Z3,  XZ2, YZ2, Z(X2 - Y2)
    # Philips: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, 3rd order not implemented
    return out_list


def reorder_to_manufacturer(spher_harm, manufacturer):
    """
    Reorder 1st - 2nd - 3rd order coefficients, if specified. From

    Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, Z2Y, Z3, Z2X, Z(X2 - Y2), X(X2 - Y2)
    (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, Z3, Z2X, Z2Y, Z(X2 - Y2) (in line with Siemens shims) or

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with GE shims) or

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, Z3, Z2X, Z2Y, Z(X2 - Y2), XYZ, X(X2 - Y2), Y(X2 - Y2) (in line with Philips shims)

    Args:
        spher_harm (dict): 3D array of spherical harmonics coefficients with key corresponding to the order
        manufacturer (str): Manufacturer of the scanner

    Returns:
        dict: Coefficients ordered following the manufacturer's convention
    """
    if manufacturer not in MANUFACTURERS:
        # Do not reorder if the manufacturer is not in the implemented manufacturers
        return spher_harm

    def _reorder_order0(sph, manuf):
        if sph.shape[-1] != 1:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 1")
        return sph[..., [0]]

    def _reorder_order1(sph, manuf):
        if sph.shape[-1] != 3:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 3")
        if manuf in ['SIEMENS', 'GE', 'PHILIPS']:
            return sph[..., [2, 0, 1]]
        else:
            logger.warning(f"1st order spherical harmonics not implemented for: {manuf}")
            return sph

    def _reorder_order2(sph, manuf):
        if sph.shape[-1] != 5:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 5")

        if manuf in ['SIEMENS', 'PHILIPS', 'GE']:
            return sph[..., [2, 3, 1, 4, 0]]
        else:
            logger.warning(f"2nd order spherical harmonics not implemented for: {manuf}")
            return sph

    def _reorder_order3(sph, manuf):
        if sph.shape[-1] != 7:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 7")
        if manufacturer == 'SIEMENS':
            return sph[..., [3, 4, 2, 5]]
        elif manufacturer == 'PHILIPS':
            # Y(X2 - Y2), XYZ, Z2Y, Z3, Z2X, Z(X2 - Y2), X(X2 - Y2)
            # Z3, Z2X, Z2Y, Z(X2 - Y2), XYZ, X(X2 - Y2), Y(X2 - Y2)
            return sph[..., [3, 4, 2, 5, 1, 6, 0]]

        else:
            logger.warning(f"3rd order spherical harmonics not implemented for: {manuf}")
            return sph

    reorder = {0: _reorder_order0,
               1: _reorder_order1,
               2: _reorder_order2,
               3: _reorder_order3}

    reordered = {}
    for order in spher_harm.keys():
        if order not in reorder.keys():
            logger.warning(f"Ordering for order {order} spherical harmonics not implemented")
        reordered[order] = reorder[order](spher_harm[order], manuf=manufacturer)

    return reordered
