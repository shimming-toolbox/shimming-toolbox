#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
from shimmingtoolbox.optimizer.basic_opt import BasicOptimizer


def sequential_zslice(unshimmed, coils, full_mask, z_slices):
    """
    Performs shimming slice by slice using shimmingtoolbox.optimizer.basic_opt.BasicOptimizer

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        coils (numpy.ndarray): Coil sensitivity profile as defined in coils.siemens_basis.siemens_basis()
        full_mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        z_slices (numpy.ndarray): 1D array containing z slices to shim

    Returns:
        numpy.ndarray: Coefficients to enter in the Syngo console (this might change in the future)
                       (coils.size x z_slices.size)

    """
    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, coils.shape[3]))
    optimizer = BasicOptimizer(coils)
    for i in range(z_slices.size):
        z = z_slices[i]
        mask = full_mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(unshimmed, mask, mask_origin=(0, 0, z))

    return currents
