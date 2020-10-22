#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from shimmingtoolbox.optimizer.lsq_optimizer import LSQ_Optimizer


def sequential_zslice(unshimmed, coils, full_mask, z_slices, bounds=None):
    """
    Performs shimming slice by slice using shimmingtoolbox.optimizer.basic_opt.BasicLSQ

    Args:
        unshimmed (numpy.ndarray): 3D B0 map
        coils (numpy.ndarray): Coil sensitivity profile as defined in coils.siemens_basis.siemens_basis()
        full_mask (numpy.ndarray): 3D mask used for the optimizer (only consider voxels with non-zero values).
        z_slices (numpy.ndarray): 1D array containing z slices to shim
        bounds (list): List of ``(min, max)`` pairs for each coil channels. None
               is used to specify no bound.
    Returns:
        numpy.ndarray: Coefficients to enter in the Syngo console (this might change in the future)
                       (coils.size x z_slices.size)

    """
    z_slices.reshape(z_slices.size)
    currents = np.zeros((z_slices.size, coils.shape[3]))
    optimizer = LSQ_Optimizer(coils)
    for i in range(z_slices.size):
        z = z_slices[i]
        mask = full_mask[:, :, z:z+1]
        currents[i, :] = optimizer.optimize(unshimmed, mask, mask_origin=(0, 0, z), bounds=bounds)

    return currents
