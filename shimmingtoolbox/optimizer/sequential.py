#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]


def sequential_zslice(unshimmed, list_coil: ListCoil, full_mask, z_slices, method='least_squares'):
    """
    Performs shimming slice by slice using one of the supported optimizers

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        list_coil (ListCoil): Coil sensitivity profile as defined in coils.siemens_basis.siemens_basis()
        full_mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
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
        optimizer = supported_optimizer[method](list_coil)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    # Count number of channels
    n_channels = 0
    for i in range(len(list_coil)):
        n_channels += list_coil[i].profiles.shape[3]

    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, n_channels))
    for i in range(z_slices.size):
        mask = np.full_like(full_mask, fill_value=False)
        z = z_slices[i]
        mask[:, :, z:z+1] = full_mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(unshimmed, mask)

    return currents
