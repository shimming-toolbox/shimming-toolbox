#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp

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


def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, pmu: PmuResp, coils: ListCoil, static_mask, riro_mask, slices,
                                opt_method='least_squares'):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries.
        json_fmap (dict): dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        pmu (PmuResp): Filename of the file of the respiratory trace.
        coils (ListCoil): List of Coils containing the coil profiles
        static_mask (numpy.ndarray): 3D mask used for the optimizer to shim the region for the static component.
        riro_mask (numpy.ndarray): 3D mask used for the optimizer to shim the region for the static component.
        slices (list): 1D array containing tuples of z slices to shim
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [hz]
            * numpy.ndarray: Static coefficients to shim (len(slices) x channels) [hz/unit_pressure]
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    affine = nii_fieldmap.affine
    if fieldmap.ndim != 4:
        raise RuntimeError("fmap must be 4d (x, y, z, t)")

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
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [hz/unit_pressure]

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine, coils)
    currents_static = optimize(optimizer, static_mask, slices)

    # Riro shim
    # We multiply by the max offset so that the bounds take effect on the maximum value that the pressure probe can
    # acquire. The equation "riro(i_vox) * (acq_pressures - mean_p)" becomes "riro(i_vox) * max_offset" which is the
    # maximum shim we will have. We solve for that to make sure the coils can support it. The units of riro * max_offset
    # are: [hz]
    max_offset = max(4095 - mean_p, mean_p)
    optimizer.set_unshimmed(riro * max_offset, affine)
    currents_max_riro = optimize(optimizer, riro_mask, slices)
    # Once the currents are solved, we divide by max_offset to return to units of [hz/unit_pressure]
    currents_riro = currents_max_riro / max_offset

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    currents_riro_rms = currents_riro * pressure_rms  # [hz/unit_pressure] * rms_pressure

    return currents_static, currents_riro, pressure_rms


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


def optimize(optimizer: Optimizer, mask, slices):
    """
        Optimize in the specified ROI according to a specified Optimizer and specified slices

    Args:
        optimizer (Optimizer): Initialized Optimizer object
        mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        slices (list): 1D array containing tuples of z slices to shim

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
        currents[i, :] = optimizer.optimize(sliced_mask)

    return currents
