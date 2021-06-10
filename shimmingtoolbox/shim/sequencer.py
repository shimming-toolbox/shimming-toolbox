#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil

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
        Shim slicewise in the specified ROI

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
