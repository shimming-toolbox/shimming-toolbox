#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression
import nibabel as nib
import logging
from nibabel.affines import apply_affine

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.masking.mask_utils import resample_mask

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

supported_optimizers = {
    'least_squares': LsqOptimizer,
    'pseudo_inverse': Optimizer
}


def shim_sequencer(nii_fieldmap, nii_anat, nii_mask_anat, slices, coils: ListCoil, method='least_squares',
                   mask_dilation_kernel='sphere', mask_dilation_kernel_size=3):
    """
    Performs shimming according to slices using one of the supported optimizers and coil profiles.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 3d and an affine transformation.
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_mask_anat (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim in the region of interest.
                                             (only consider voxels with non-zero values)
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape of anat
                       is: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil profiles
                          are larger than the extent of the fieldmap. This is especially true for dimensions with only
                          1 voxel(e.g. (50x50x1). Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                          :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'. Note: refer to their specific
                      implementation to know limits of the methods in: :mod:`shimmingtoolbox.optimizer`
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.

    Returns:
        numpy.ndarray: Coefficients of the coil profiles to shim (len(slices) x n_channels)
    """

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fieldmap.get_fdata().ndim != 3:
        raise ValueError("Fieldmap must be 3d (dim1, dim2, dim3)")
    fieldmap_shape = nii_fieldmap.get_fdata().shape
    # Extend the fieldmap if there are axes that are 1d
    if 1 in fieldmap_shape:
        list_axis = [i for i in range(len(fieldmap_shape)) if fieldmap_shape[i] == 1]
        n_slices = int(math.ceil((mask_dilation_kernel_size - 1) / 2))
        for i_axis in list_axis:
            nii_fieldmap = extend_slice(nii_fieldmap, n_slices=n_slices, axis=i_axis)
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise ValueError("Anatomical image must be in 3d")

    # Make sure the mask has the appropriate dimensions
    mask = nii_mask_anat.get_fdata()
    if mask.ndim != 3:
        raise ValueError("Mask image must be in 3d")

    # Make sure shape and affine of mask are the same as the anat
    if not np.all(mask.shape == anat.shape):
        raise ValueError(f"Shape of mask:\n {mask.shape} must be the same as the shape of anat:\n{anat.shape}")
    if not np.all(nii_mask_anat.affine == nii_anat.affine):
        raise ValueError(f"Affine of mask:\n{nii_mask_anat.affine}\nmust be the same as the affine of anat:\n"
                         f"{nii_anat.affine}")

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, fieldmap, affine_fieldmap, coils)

    # Optimize slice by slice
    coef = _optimize(optimizer, nii_mask_anat, slices, dilation_kernel=mask_dilation_kernel,
                     dilation_size=mask_dilation_kernel_size)

    return coef


def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices,
                                pmu: PmuResp, coils: ListCoil, opt_method='least_squares',
                                mask_dilation_kernel='sphere', mask_dilation_kernel_size=3):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries. Also contains an affine transformation.
        json_fmap (dict): Dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_static_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                               component.
        nii_riro_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                             component.
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat where the shape of anat:
                       (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        pmu (PmuResp): PmuResp object containing the respiratory trace information.
        coils (ListCoil): List of `Coils` containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil profiles
                          are larger than the extent of the fieldmap. This is especially true for dimensions with only
                          1 voxel(e.g. (50x50x1x10). Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                          :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'. Note: refer to their specific
                          implementation to know limits of the methods in: :mod:`shimmingtoolbox.optimizer`
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients of the coil profiles to shim (len(slices) x channels) e.g. [Hz]
            * numpy.ndarray: Riro coefficients of the coil profiles to shim (len(slices) x channels)
                             e.g. [Hz/unit_pressure]
            * float: Mean pressure of the respiratory trace.
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """
    # Note: We technically dont need the anat if we use the nii_mask. However, this is a nice safety check to make sure
    # the mask is indeed in the dimension of the anat and not the fieldmap.

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fieldmap.get_fdata().ndim != 4:
        raise ValueError("Fieldmap must be 4d (dim1, dim2, dim3, t)")
    fieldmap_shape = nii_fieldmap.get_fdata().shape[:3]
    # Extend the fieldmap if there are axes that are 1d
    if 1 in fieldmap_shape:
        list_axis = [i for i in range(len(fieldmap_shape)) if fieldmap_shape[i] == 1]
        for i_axis in list_axis:
            n_slices = int(math.ceil((mask_dilation_kernel_size - 1) / 2))
            nii_fieldmap = extend_slice(nii_fieldmap, n_slices=n_slices, axis=i_axis)
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise ValueError("Anatomical image must be in 3d")

    # Make sure masks have the appropriate dimensions
    static_mask = nii_static_mask.get_fdata()
    if static_mask.ndim != 3:
        raise ValueError("static_mask image must be in 3d")
    riro_mask = nii_riro_mask.get_fdata()
    if riro_mask.ndim != 3:
        raise ValueError("riro_mask image must be in 3d")

    # Make sure shape and affine of masks are the same as the anat
    if not (np.all(riro_mask.shape == anat.shape) and np.all(static_mask.shape == anat.shape)):
        raise ValueError(f"Shape of riro mask: {riro_mask.shape} and static mask: {static_mask.shape} "
                         f"must be the same as the shape of anat: {anat.shape}")
    if not(np.all(nii_riro_mask.affine == nii_anat.affine) and np.all(nii_static_mask.affine == nii_anat.affine)):
        raise ValueError(f"Affine of riro mask:\n{nii_riro_mask.affine}\nand static mask: {nii_static_mask.affine}\n"
                         f"must be the same as the affine of anat:\n{nii_anat.affine}")

    # Fetch PMU timing
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_fmap)
    # TODO: deal with saturation
    # fit PMU and fieldmap values
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # regularization --> static, riro
    # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
    mean_p = np.mean(acq_pressures)
    pressure_rms = np.sqrt(np.mean((acq_pressures - mean_p) ** 2))
    reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p, fieldmap.reshape(-1, fieldmap.shape[-1]).T)
    # TODO: Safety check for linear regression if the pressure and fieldmap fits

    # static/riro contains a 3d matrix of static/riro map in the fieldmap space considering the previous equation
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine_fieldmap, coils)
    coef_static = _optimize(optimizer, nii_static_mask, slices, dilation_kernel=mask_dilation_kernel,
                            dilation_size=mask_dilation_kernel_size)

    # Use the currents to define a list of new coil bounds for the riro optimization
    bounds = new_bounds_from_currents(coef_static, optimizer.merged_bounds)

    # Riro shim
    # We multiply by the max offset of the siemens pmu e.g. [max - min = 4095] so that the bounds take effect on the
    # maximum value that the pressure probe can acquire. The equation "riro(i_vox) * (acq_pressures - mean_p)" becomes
    # "riro(i_vox) * max_offset" which is the maximum riro shim we will have. We solve for that to make sure the coils
    # can support it. The units of riro * max_offset are: [unit_shim], ex: [Hz]
    max_offset = max((pmu.max - pmu.min) - mean_p, mean_p)

    # Set the riro map to shim
    # TODO: make sure max_offset could not bust with negative offset
    optimizer.set_unshimmed(riro * max_offset, affine_fieldmap)
    coef_max_riro = _optimize(optimizer, nii_riro_mask, slices, shimwise_bounds=bounds,
                              dilation_kernel=mask_dilation_kernel, dilation_size=mask_dilation_kernel_size)
    # Once the coefficients are solved, we divide by max_offset to return to units of
    # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]
    coef_riro = coef_max_riro / max_offset

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    # coef_riro_rms = coef_riro * pressure_rms
    # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

    return coef_static, coef_riro, mean_p, pressure_rms


def new_bounds_from_currents(currents, old_bounds):
    """
    Uses the currents to determine the appropriate bounds for the next optimization. It assumes that
    "old_coef + next_bound < old_bound".

    Args:
        currents (np.ndarray): 2D array (n_shims x n_channels). Direct output from :func:`_optimize`.
        old_bounds (list): 1d list (n_channels) of tuples (min, max) containing the merged bounds of the previous
                           optimization.

    Returns:
        list: 2d list (n_shims x n_channels) of bounds (min, max) corresponding to each shim and channel.
    """

    new_bounds = []
    for i_shim in range(currents.shape[0]):
        shim_bound = []
        for i_channel in range(len(old_bounds)):
            a_bound = old_bounds[i_channel] - currents[i_shim, i_channel]
            shim_bound.append(tuple(a_bound))
        new_bounds.append(shim_bound)

    return new_bounds


def select_optimizer(method, unshimmed, affine, coils: ListCoil):
    """
    Select and initialize the optimizer

    Args:
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'
        unshimmed (numpy.ndarray): 3D B0 map
        affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        coils (ListCoil): List of Coils containing the coil profiles

    Returns:
        Optimizer: Initialized Optimizer object
    """

    # global supported_optimizers
    if method in supported_optimizers:
        optimizer = supported_optimizers[method](coils, unshimmed, affine)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    return optimizer


def _optimize(optimizer: Optimizer, nii_mask_anat, slices_anat, shimwise_bounds=None,
              dilation_kernel='sphere', dilation_size=3):

    # Count number of channels
    n_channels = optimizer.merged_coils.shape[3]

    # Count shims to perform
    n_shims = len(slices_anat)

    # Initialize
    coefs = np.zeros((n_shims, n_channels))

    # For each shim
    for i in range(n_shims):
        # Create nibabel object of the unshimmed map
        nii_unshimmed = nib.Nifti1Image(optimizer.unshimmed, optimizer.unshimmed_affine)

        # Create mask in the fieldmap coordinate system from the anat roi mask and slice anat mask
        sliced_mask_resampled = resample_mask(nii_mask_anat, nii_unshimmed, slices_anat[i],
                                              dilation_kernel=dilation_kernel, dilation_size=dilation_size).get_fdata()

        # If new bounds are included, change them for each shim
        if shimwise_bounds is not None:
            optimizer.set_merged_bounds(shimwise_bounds[i])

        # Optimize using the mask
        coefs[i, :] = optimizer.optimize(sliced_mask_resampled)

    return coefs


def update_affine_for_ap_slices(affine, n_slices=1, axis=2):
    """ Updates the input affine to reflect an insertion of n_slices on each side of the selected axis

    Args:
        affine (numpy.ndarray): 4x4 qform affine matrix representing the coordinates
        n_slices (int): Number of pixels to add on each side of the selected axis
        axis (int): Axis along which to insert the slice(s)

    Returns:
        (numpy.ndarray): 4x4 updated affine matrix
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


def extend_slice(nii_array, n_slices=1, axis=2):
    """ Adds n_slices on each side of the selected axis. It uses the nearest slice and copies it to fill the values.
    Updates the affine of the matrix to keep the input array in the same location.

    Args:
        nii_array (nib.Nifti1Image): 3d or 4d array to extend the dimensions along an axis.
        n_slices (int): Number of slices to add on each side of the selected axis.
        axis (int): Axis along which to insert the slice(s).

    Returns:
        nib.Nifti1Image: Array extended with the appropriate affine to conserve where the original pixels were located.

    Examples:

        ::

            print(nii_array.get_fdata().shape)  # (50, 50, 1, 10)
            nii_out = extend_slice(nii_array, n_slices=1, axis=2)
            print(nii_out.get_fdata().shape)  # (50, 50, 3, 10)

    """
    if nii_array.get_fdata().ndim == 3:
        extended = nii_array.get_fdata()
        extended = extended[..., np.newaxis]
    elif nii_array.get_fdata().ndim == 4:
        extended = nii_array.get_fdata()
    else:
        raise ValueError("Unsupported number of dimensions for input array")

    for i_slice in range(n_slices):
        if axis == 0:
            extended = np.insert(extended, -1, extended[-1, :, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[0, :, :, :], axis=axis)
        elif axis == 1:
            extended = np.insert(extended, -1, extended[:, -1, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, 0, :, :], axis=axis)
        elif axis == 2:
            extended = np.insert(extended, -1, extended[:, :, -1, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, :, 0, :], axis=axis)
        else:
            raise ValueError("Unsupported value for axis")

    new_affine = update_affine_for_ap_slices(nii_array.affine, n_slices, axis)

    if nii_array.get_fdata().ndim == 3:
        extended = extended[..., 0]

    nii_extended = nib.Nifti1Image(extended, new_affine, header=nii_array.header)

    return nii_extended


def define_slices(n_slices: int, factor=1, method='sequential'):
    """
    Define the slices to shim according to the output convention. (list of tuples)

    Args:
        n_slices (int): Number of total slices.
        factor (int): Number of slices per shim.
        method (str): Defines how the slices should be sorted, supported methods include: 'interleaved', 'sequential',
                      'volume'. See Examples for more details.

    Returns:
        list: 1D list containing tuples of dim3 slices to shim. (dim1, dim2, dim3)

    Examples:

        ::

            slices = define_slices(10, 2, 'interleaved')
            print(slices)  # [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

            slices = define_slices(20, 5, 'sequential')
            print(slices)  # [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)]

            slices = define_slices(20, method='volume')
            print(slices)  # [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]

    """
    if n_slices <= 0:
        raise ValueError("Number of slices should be greater than 0")

    slices = []
    n_shims = n_slices // factor
    leftover = n_slices % factor

    if method == 'interleaved':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

    elif method == 'sequential':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim * factor, (i_shim + 1) * factor, 1)))

    elif method == 'volume':
        slices.append(tuple(range(n_shims)))

    else:
        raise ValueError("Not a supported method to define slices")

    if leftover != 0:
        slices.append(tuple(range(n_shims * factor, n_slices)))
        logger.warning(f"When defining the slices to shim, there are leftover slices since the factor used and number "
                       f"of slices is not perfectly dividable. Make sure the last tuple of slices is "
                       f"appropriate: {slices}")

    return slices
