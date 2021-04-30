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
        numpy.ndarray: Coefficients to enter in the Syngo console (this might change in the future)
                       (coils.size x z_slices.size)

    """

    supported_optimizer = {
        'least_squares': LsqOptimizer,
        'pseudo_inverse': Optimizer
    }

    if method in supported_optimizer:
        optimizer = supported_optimizer[method](coils)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    # Count number of channels
    n_channels = 0
    for i in range(len(coils)):
        n_channels += coils[i].profile.shape[3]

    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, n_channels))
    for i in range(z_slices.size):
        sliced_mask = np.full_like(mask, fill_value=False)
        z = z_slices[i]
        sliced_mask[:, :, z:z+1] = mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(unshimmed, affine, sliced_mask)

    return currents
