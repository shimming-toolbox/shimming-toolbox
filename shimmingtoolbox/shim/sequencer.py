#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_opening
import nibabel as nib
import logging

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coordinates import resample_from_to

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

supported_optimizers = {
    'least_squares': LsqOptimizer,
    'pseudo_inverse': Optimizer
}


def shim_sequencer(nii_fieldmap, nii_anat, nii_mask_anat, slices, coils: ListCoil, method='least_squares'):
    """
    Performs shimming according to slices using one of the supported optimizers and coil profiles.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 3d.
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_mask_anat (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim in the region of interest.
                                             (only consider voxels with non-zero values)
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape of anat
                       is: (dim1, dim2, dim3). Refer to shimmingtoolbox.shim.sequencer:define_slices().
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to shimmingtoolbox.coils.coil:Coil().
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'. Note: refer to their specific
                      implementation to know limits of the methods in: shimmingtoolbox.optimizer

    Returns:
        numpy.ndarray: Coefficients to shim (len(slices) x channels)
    """

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine
    if fieldmap.ndim != 3:
        raise ValueError("Fieldmap must be 3d (dim1, dim2, dim3)")

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
        raise ValueError(f"Shape of mask: {mask.shape} must be the same as the shape of anat: {anat.shape}")
    if not np.all(nii_mask_anat.affine == nii_anat.affine):
        raise ValueError(f"Affine of mask: {nii_mask_anat.affine} must be the same as the affine of anat: "
                         f"{nii_anat.affine}")

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, fieldmap, affine_fieldmap, coils)

    # Optimize slice by slice
    currents = optimize(optimizer, nii_mask_anat, slices)

    return currents


def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices,
                                pmu: PmuResp, coils: ListCoil, opt_method='least_squares'):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries.
        json_fmap (dict): Dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_static_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                               component.
        nii_riro_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                             component.
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat where the shape of anat:
                       (dim1, dim2, dim3). Refer to shimmingtoolbox.shim.sequencer:define_slices().
        pmu (PmuResp): Filename of the file of the respiratory trace.
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'.

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) e.g. [Hz]
            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) e.g. [Hz/unit_pressure]
            * float: Mean pressure of the respiratory trace.
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """
    # Note: We technically dont need the anat if we use the nii_mask. However, this is a nice safety check to make sur
    # the mask is indeed in the dimension of the anat and not the fieldmap

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine
    if fieldmap.ndim != 4:
        raise RuntimeError("Fieldmap must be 4d (dim1, dim2, dim3, t)")

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise RuntimeError("Anatomical image must be in 3d")

    # Make sure masks have the appropriate dimensions
    static_mask = nii_static_mask.get_fdata()
    if static_mask.ndim != 3:
        raise RuntimeError("static_mask image must be in 3d")
    riro_mask = nii_riro_mask.get_fdata()
    if riro_mask.ndim != 3:
        raise RuntimeError("riro_mask image must be in 3d")

    # Make sure shape and affine of masks are the same as the anat
    if not (np.all(riro_mask.shape == anat.shape) and np.all(static_mask.shape == anat.shape)):
        raise ValueError(f"Shape of riro mask: {riro_mask.shape} and static mask: {static_mask.shape} "
                         f"must be the same as the shape of anat: {anat.shape}")
    if not(np.all(nii_riro_mask.affine == nii_anat.affine) and np.all(nii_static_mask.affine == nii_anat.affine)):
        raise ValueError(f"Affine of riro mask: {nii_riro_mask.affine} and static mask: {nii_static_mask.affine} "
                         f"must be the same as the affine of anat: {nii_anat.affine}")

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

    # static/riro contains a 3d matrix of static/riro coefficients in the fieldmap space
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine_fieldmap, coils)
    currents_static = optimize(optimizer, nii_static_mask, slices)

    # Use the currents to define a list of new bounds for the riro optimization
    bounds = new_bounds_from_currents(currents_static, optimizer.merged_bounds)

    # Riro shim
    # We multiply by the max offset of the siemens pmu [max - min = 4095] so that the bounds take effect on the maximum
    # value that the pressure probe can acquire. The equation "riro(i_vox) * (acq_pressures - mean_p)" becomes
    # "riro(i_vox) * max_offset" which is the maximum shim we will have. We solve for that to make sure the coils can
    # support it. The units of riro * max_offset are: [unit_shim], ex: [Hz]
    max_offset = max((pmu.max - pmu.min) - mean_p, mean_p)

    # Set the riro map to shim
    optimizer.set_unshimmed(riro * max_offset, affine_fieldmap)
    currents_max_riro = optimize(optimizer, nii_riro_mask, slices, shimwise_bounds=bounds)
    # Once the currents are solved, we divide by max_offset to return to units of
    # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]
    currents_riro = currents_max_riro / max_offset

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    # currents_riro_rms = currents_riro * pressure_rms
    # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

    return currents_static, currents_riro, mean_p, pressure_rms


def new_bounds_from_currents(currents, old_bounds):
    """
    Uses the currents to determine the appropriate bound for a next optimization. It assumes that
    "old_current + next_current < old_bound".

    Args:
        currents: 2D array (n_shims x n_channels). Direct output from ``optimize``.
        old_bounds: 1d list (n_channels) of the merged bounds of the previous optimization.

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


def optimize(optimizer: Optimizer, nii_mask_anat, slices_anat, shimwise_bounds=None):

    # Count number of channels
    n_channels = optimizer.merged_coils.shape[3]

    # Count shims to perform
    n_shims = len(slices_anat)

    # Initialize
    currents = np.zeros((n_shims, n_channels))

    # For each shim
    for i in range(n_shims):
        # Create nibabel object of the unshimmed map
        nii_unshimmed = nib.Nifti1Image(optimizer.unshimmed, optimizer.unshimmed_affine)

        # Create mask in the fieldmap coordinate system from the anat roi mask and slice anat mask
        sliced_mask_resampled = resample_mask(nii_mask_anat, nii_unshimmed, slices_anat[i]).get_fdata()

        # If new bounds are included, change them for each shim
        if shimwise_bounds is not None:
            optimizer.set_merged_bounds(shimwise_bounds[i])

        # Optimize using the mask
        currents[i, :] = optimizer.optimize(sliced_mask_resampled)

    return currents


def resample_mask(nii_mask_from, nii_target, from_slices):
    """
    Select the appropriate slices from ``nii_mask_from`` using ``from_slices`` and resample onto ``nii_target``

    Args:
        nii_mask_from (nib.Nifti1Image): Mask to resample from. False or 0 signifies not included.
        nii_target (nib.Nifti1Image): Target image to resample onto.
        from_slices (tuple): Tuple containing the slices to select from nii_mask_from.

    Returns:
        nib.Nifti1Image: Mask resampled with nii_target.shape and nii_target.affine.
    """

    mask_from = nii_mask_from.get_fdata()

    # Initialize a sliced mask and select the slices from from_slices
    sliced_mask = np.full_like(mask_from, fill_value=False)
    sliced_mask[:, :, from_slices] = mask_from[:, :, from_slices]

    # Create nibabel object
    nii_mask = nib.Nifti1Image(sliced_mask.astype(int), nii_mask_from.affine, header=nii_mask_from.header)

    # Resample the mask onto nii_target
    nii_mask_target = resample_from_to(nii_mask, nii_target, order=0, mode='grid-constant', cval=0)

    # TODO: Add pixels/slices if the number of pixel is too small in a direction, dilation?
    # Straight up dilation won't work since it will add pixels in every direction regardless
    def dilate_mask(mask, n_pixels, direction='all'):
        if direction == 'all':
            return binary_dilation(mask)
        elif direction == 'individual':

            # TODO: remove
            mask[5,5] = 1

            # TODO: use n_pixels to dilate an appropriate amount of pixels
            struct_dim1 = np.zeros([3, 3, 3])
            struct_dim1[:, 1, 1] = 1
            # Finds where the structure fits
            open1 = binary_opening(mask, structure=struct_dim1)
            # Select Everything that does not fit within the structure and erode along a dim
            dim1 = binary_dilation(np.logical_and(np.logical_not(open1), mask), structure=struct_dim1)

            struct_dim2 = np.zeros([3, 3, 3])
            struct_dim2[1, :, 1] = 1
            # Finds where the structure fits
            open2 = binary_opening(mask, structure=struct_dim2)
            # Select Everything that does not fit within the structure and erode along a dim
            dim2 = binary_dilation(np.logical_and(np.logical_not(open2), mask), structure=struct_dim2)

            struct_dim3 = np.zeros([3, 3, 3])
            struct_dim3[1, 1, :] = 1
            # Finds where the structure fits
            open3 = binary_opening(mask, structure=struct_dim3)
            # Select Everything that does not fit within the structure and erode along a dim
            dim3 = binary_dilation(np.logical_and(np.logical_not(open3), mask), structure=struct_dim3)

            mask_dilated = np.logical_or(np.logical_or(np.logical_or(dim1, dim2), dim3), mask)

            return mask_dilated.astype(int)

    mask_dilated = dilate_mask(nii_mask_target.get_fdata(), 1, 'individual')
    mask_dilated = dilate_mask(mask_dilated, 1, 'individual')
    nii_mask_dilated = nib.Nifti1Image(mask_dilated, nii_mask_target.affine, header=nii_mask_target.header)

    #######
    # Debug TODO: REMOVE
    import os
    nib.save(nii_mask, os.path.join(os.curdir, f"fig_mask_{from_slices[0]}.nii.gz"))
    nib.save(nii_mask_from, os.path.join(os.curdir, "fig_mask_roi.nii.gz"))
    nib.save(nii_mask_target, os.path.join(os.curdir, f"fig_mask_res{from_slices[0]}.nii.gz"))
    nib.save(nii_mask_dilated, os.path.join(os.curdir, f"fig_mask_dilated{from_slices[0]}.nii.gz"))
    #######

    return nii_mask_target


def define_slices(n_slices: int, factor: int, method='interleaved'):
    """
    Define the slices to shim according to the output convention.

    Args:
        n_slices (int): Number of total slices.
        factor (int): Number of slices per shim.
        method (str): Defines how the slices should be sorted, supported methods include: 'interleaved', 'sequential'.
                      See Examples for more details.

    Returns:
        list: 1D list containing tuples of z slices to shim.

    Examples:

        ::

            slices = define_slices(10, 2, 'interleaved')
            print(slices)  # [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

            slices = define_slices(20, 5, 'sequential')
            print(slices)  # [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)]

    """
    if n_slices <= 0:
        return [tuple()]

    slices = []
    n_shims = n_slices // factor
    leftover = n_slices % factor

    if method == 'interleaved':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

    elif method == 'sequential':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim * factor, (i_shim + 1) * factor, 1)))

    else:
        raise NotImplementedError("Not a supported method to define slices")

    if leftover != 0:
        slices.append(tuple(range(n_shims * factor, n_slices)))
        logger.warning(f"When defining the slices to shim, there are leftover slices since the factor used and number "
                       f"of slices is not perfectly dividable. Make sure the last tuple of slices is "
                       f"appropriate: {slices}")

    return slices
