#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression
import nibabel as nib
import os

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coordinates import resample_from_to

ListCoil = List[Coil]

supported_optimizers = {
    'least_squares': LsqOptimizer,
    'pseudo_inverse': Optimizer
}


def shim_sequencer(unshimmed, affine, coils: ListCoil, mask, slices, method='least_squares'):
    """
    Performs shimming slice by slice using one of the supported optimizers

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        coils (ListCoil): List of Coils containing the coil profiles
        mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        slices (list): 1D array containing tuples of z slices to shim
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'

    Returns:
        numpy.ndarray: Coefficients to shim (len(slices) x channels)
    """

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, unshimmed, affine, coils)

    # Optimize slice by slice
    currents = optimize(optimizer, mask, slices)

    return currents


def shim_realtime_pmu_sequencer_anat(nii_fieldmap, json_fmap, nii_anat, pmu: PmuResp, coils: ListCoil, nii_static_mask,
                                   nii_riro_mask, slices, opt_method='least_squares'):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries.
        json_fmap (dict): dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        pmu (PmuResp): Filename of the file of the respiratory trace.
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
        nii_static_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                               component.
        nii_riro_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                             component.
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat. (dim1, dim2, dim3)
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'.

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [Hz]
            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [Hz/unit_pressure]
            * float: Mean pressure of the respiratory trace.
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """
    # We technically dont need the anat if we use the nii_mask

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine
    if fieldmap.ndim != 4:
        raise RuntimeError("Fieldmap must be 4d (x, y, z, t)")

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
    currents_static = optimize_from_anat(optimizer, nii_static_mask, slices)

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
    currents_max_riro = optimize_from_anat(optimizer, nii_riro_mask, slices, shimwise_bounds=bounds)
    # Once the currents are solved, we divide by max_offset to return to units of
    # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]
    currents_riro = currents_max_riro / max_offset

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    # currents_riro_rms = currents_riro * pressure_rms
    # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

    return currents_static, currents_riro, mean_p, pressure_rms


def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, pmu: PmuResp, coils: ListCoil, static_mask, riro_mask, slices,
                                opt_method='least_squares'):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries.
        json_fmap (dict): dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        pmu (PmuResp): Filename of the file of the respiratory trace.
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
        static_mask (numpy.ndarray): 3D mask used for the optimizer to shim the region for the static component.
        riro_mask (numpy.ndarray): 3D mask used for the optimizer to shim the region for the riro component.
        slices (list): 1D array containing tuples of z slices to shim.
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'.

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [Hz]
            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [Hz/unit_pressure]
            * float: Mean pressure of the respiratory trace.
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    affine = nii_fieldmap.affine
    if fieldmap.ndim != 4:
        raise RuntimeError("Fieldmap must be 4d (x, y, z, t)")

    # Fetch PMU timing
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_fmap)
    # TODO: deal with saturation
    # fit PMU and fieldmap values
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)
    # # DEBUG:
    # acq_pressures = np.array([3000, 2000, 1000, 2000])

    # regularization --> static, riro
    # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
    mean_p = np.mean(acq_pressures)
    pressure_rms = np.sqrt(np.mean((acq_pressures - mean_p) ** 2))
    reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p, fieldmap.reshape(-1, fieldmap.shape[-1]).T)

    # static/riro contains a 3d matrix of static/riro coefficients
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine, coils)
    currents_static = optimize(optimizer, static_mask, slices)

    # Use the currents to define a list of new bounds for the riro optimization
    bounds = new_bounds_from_currents(currents_static, optimizer.merged_bounds)

    # Riro shim
    # We multiply by the max offset of the siemens pmu [max - min = 4095] so that the bounds take effect on the maximum
    # value that the pressure probe can acquire. The equation "riro(i_vox) * (acq_pressures - mean_p)" becomes
    # "riro(i_vox) * max_offset" which is the maximum shim we will have. We solve for that to make sure the coils can
    # support it. The units of riro * max_offset are: [unit_shim], ex: [Hz]
    max_offset = max((pmu.max - pmu.min) - mean_p, mean_p)

    # Set the riro map to shim
    optimizer.set_unshimmed(riro * max_offset, affine)
    currents_max_riro = optimize(optimizer, riro_mask, slices, shimwise_bounds=bounds)
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


def optimize_from_anat(optimizer: Optimizer, nii_mask_anat, slices_anat, shimwise_bounds=None):

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


def optimize(optimizer: Optimizer, mask, slices, shimwise_bounds=None):
    """
    Optimize in the specified ROI according to a specified Optimizer and specified slices

    Args:
        optimizer (Optimizer): Initialized Optimizer object
        mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        slices (list): 1D array containing tuples of z slices to shim
        shimwise_bounds (list): list (n shims) of list (n channels) of tuples (min_coef, max_coef)

    Returns:
        numpy.ndarray: Coefficients to shim (len(slices) x channels)
    """
    # Count number of channels
    n_channels = optimizer.merged_coils.shape[3]
    n_shims = len(slices)
    currents = np.zeros((n_shims, n_channels))
    for i in range(n_shims):
        sliced_mask = np.full_like(mask, fill_value=False)
        sliced_mask[:, :, slices[i]] = mask[:, :, slices[i]]
        if shimwise_bounds is not None:
            optimizer.set_merged_bounds(shimwise_bounds[i])
        currents[i, :] = optimizer.optimize(sliced_mask)

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

    # TODO: Add pixels/slices if the number of pixel is too small in a direction

    #######
    # Debug
    nib.save(nii_mask, os.path.join(os.curdir, f"fig_mask_{from_slices[0]}.nii.gz"))
    nib.save(nii_mask_from, os.path.join(os.curdir, "fig_mask_roi.nii.gz"))
    nib.save(nii_mask_target, os.path.join(os.curdir, f"fig_mask_res{from_slices[0]}.nii.gz"))
    #######

    return nii_mask_target


def define_slices(n_slices: int, factor: int, method='separated'):
    """
    Define the slices to shim according to the output covention.

    Args:
        n_slices (int): Number of total slices.
        factor (int): Number of slices per shim.
        method (str): Defines how the slices should be seperated, supported methods include: 'separated', 'sequential'.
                      See Examples for more details.

    Returns:
        list: 1D list containing tuples of z slices to shim.

    Examples:

        ::

            slices = define_slices(10, 2, 'separated')
            print(slices)  # [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

            slices = define_slices(20, 5, 'sequential')
            print(slices)  # [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)]

    """
    if n_slices <= 0:
        return [tuple()]

    slices = []
    n_shims = n_slices // factor
    leftover = n_slices % factor

    if method == 'separated':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

    elif method == 'sequential':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim * factor, (i_shim + 1) * factor, 1)))

    else:
        raise NotImplementedError("Not a supported method to define slices")

    if leftover != 0:
        slices.append(tuple(range(n_shims * factor, n_slices)))

    return slices
