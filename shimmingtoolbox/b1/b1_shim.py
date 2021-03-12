#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import scipy.optimize


def b1_shim(b1_maps, mask):
    """
    Computes optimized shim weights that minimize the coefficient of variation of the B1 field in the masked region.

    Args:
        b1_maps (numpy.ndarray): 4D array (x, y, slice, coil) corresponding to the measured B1 field.
        mask (numpy.ndarray): 3D array (x, y, slice) corresponding to the region where shimming will be performed.

    Returns:
        numpy.ndarray: Optimized shimming weights.

    """

    x, y, n_slices, n_coils = b1_maps.shape
    b1_roi = np.reshape(b1_maps, [x*y*n_slices, n_coils])[np.reshape(mask, x*y*n_slices), :]

    # TODO: add possibility to input CP coefficients
    weights_init = np.concatenate((np.ones(n_coils), np.zeros(n_coils)))

    print(f'Coefficient of variation before shimming: {cov(combine_maps(b1_roi, vector_to_complex(weights_init)))}')

    def cost(weights):
        return cov(combine_maps(b1_roi, vector_to_complex(weights)))

    shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init).x)

    print(f'Shim coefficient: {shim_weights}')
    print(f'Coefficient of variation after shimming: {cov(combine_maps(b1_roi, shim_weights))}')
    return shim_weights


def combine_maps(b1_maps, weights):
    """
    Combines the B1 field distribution of several coils into one map representing the total B1 field magnitude.

    Args:
        b1_maps (numpy.ndarray): Measured B1 field for different coils. Last dimension must correspond to n_coils.
        weights (numpy.ndarray): 1D complex array of length n_coils.

    Returns:

    """
    return abs(np.sum(np.multiply(b1_maps, weights), b1_maps.ndim-1))


def cov(array):
    """
    Computes the coefficient of variation (CoV) of a given array.

    Args:
        array (numpy.ndarray): N-dimensional array.

    Returns:
        float: Coefficient of variation of the input array (standard deviation/mean).

    """
    return np.std(array)/np.mean(array)


def vector_to_complex(weights):
    """
    Combines real and imaginary values contained in a vector into a half long complex vector.

    Args:
        weights (numpy.ndarray): 1D array of length 2*n_coils. First/second half: real/imaginary part of shim weights.

    Returns:
        numpy.ndarray: 1D complex array of length n_coils.

    """
    return weights[:len(weights)//2] + 1j * weights[len(weights)//2:]
