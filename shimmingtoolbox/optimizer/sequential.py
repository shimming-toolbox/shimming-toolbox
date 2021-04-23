#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.coils.coil import Coil


def sequential_zslice(unshimmed, coil: Coil, full_mask, z_slices):
    """
    Performs shimming slice by slice using shimmingtoolbox.optimizer.LsqOptimizer

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        coil (Coil): Coil sensitivity profile as defined in coils.siemens_basis.siemens_basis()
        full_mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        z_slices (numpy.ndarray): 1D array containing z slices to shim

    Returns:
        numpy.ndarray: Coefficients to enter in the Syngo console (this might change in the future)
                       (coils.size x z_slices.size)

    """
    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, coil.profiles.shape[3]))
    optimizer = LsqOptimizer(coil)
    for i in range(z_slices.size):
        mask = np.full_like(full_mask, fill_value=False)
        z = z_slices[i]
        mask[:, :, z:z+1] = full_mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(unshimmed, mask)

    return currents
