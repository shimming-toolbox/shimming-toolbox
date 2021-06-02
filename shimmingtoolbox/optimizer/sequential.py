#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]


def sequential_zslice(unshimmed, affine, coils: ListCoil, mask, z_slices, method='least_squares'):
    """
    Performs shimming slice by slice using one of the supported optimizers

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        coils (ListCoil): List of Coils containing the coil profiles
        mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        z_slices (numpy.ndarray): 1D array containing z slices to shim
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'
    Returns:
        numpy.ndarray: Coefficients to shim (channels x z_slices.size)
    """

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, unshimmed, affine, coils)

    # Optimize slice by slice
    currents = optimize_slicewise(optimizer, mask, z_slices)

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
    supported_optimizer = {
        'least_squares': LsqOptimizer,
        'pseudo_inverse': Optimizer
    }

    if method in supported_optimizer:
        optimizer = supported_optimizer[method](coils, unshimmed, affine)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    return optimizer


def optimize_slicewise(optimizer: Optimizer, mask, z_slices):
    """
        Shim slicewise in the specified ROI

    Args:
        optimizer (Optimizer): Initialized Optimizer object
        mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        z_slices (numpy.ndarray): 1D array containing z slices to shim

    Returns:
        numpy.ndarray: Coefficients to shim (channels x z_slices.size)
    """
    # Count number of channels
    n_channels = optimizer.merged_coils.shape[3]

    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, n_channels))
    for i in range(z_slices.size):
        sliced_mask = np.full_like(mask, fill_value=False)
        z = z_slices[i]
        sliced_mask[:, :, z:z+1] = mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(sliced_mask)

    return currents
