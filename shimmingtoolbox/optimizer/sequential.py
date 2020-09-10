#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from shimmingtoolbox.optimizer.least_squares import LeastSquares


def sequential_zslice(unshimmed, coils, full_mask, z_slices):
    z_slices.reshape((z_slices.size, 1))
    currents = np.zeros(z_slices.size, coils.shape[3])
    optimizer = LeastSquares(coils)
    for i in range(z_slices.size):
        z = z_slices[i]
        currents[i] = optimizer.optimize(unshimmed, full_mask[:, :, z:z+1], mask_origin=(0, 0, z))
    return currents


print("test")
