#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import scipy.optimize
import warnings


def b1_shim(b1_maps, mask, cp_weights=None):
    """
    Computes static optimized shim weights that minimize the B1 field coefficient of variation over the masked region.

    Args:
        b1_maps (numpy.ndarray): 4D array (x, y, slice, coil) corresponding to the measured B1 field.
        mask (numpy.ndarray): 3D array (x, y, slice) corresponding to the region where shimming will be performed.
        cp_weights: 1D array of complex weights corresponding to the CP mode of the coil. Must be normalized.

    Returns:
        numpy.ndarray: Optimized shimming weights.

    """
    if b1_maps.ndim == 4:
        x, y, n_slices, n_coils = b1_maps.shape
    else:
        raise ValueError("Unexpected negative magnitude values.")

    if b1_maps.shape[:-1] == mask.shape:
        b1_roi = np.reshape(b1_maps, [x * y * n_slices, n_coils])[np.reshape(mask, x * y * n_slices), :]
    else:
        raise ValueError("Mask and maps dimensions not matching.")

    if cp_weights:
        if len(cp_weights) == b1_maps.shape[-1]:
            if np.isclose(np.linalg.norm(cp_weights), 1):
                weights_init = complex_to_vector(cp_weights)
            else:
                warnings.warn("Normalizing the CP mode weights.")
                weights_init = complex_to_vector(cp_weights/np.linalg.norm(cp_weights))
        else:
            raise ValueError("CP mode and maps dimensions not matching.")
    else:
        # TODO: add function to compute CP coefficients
        # Approximation of a circular polarisation to initialize the optimization
        weights_init = np.concatenate((np.ones(n_coils) / np.linalg.norm(np.ones(n_coils)),
                                       np.linspace(0, 2 * np.pi - 2 * np.pi / n_coils, n_coils)))

    print(f"Coefficient of variation before shimming: {cov(combine_maps(b1_roi, vector_to_complex(weights_init)))}")

    bounds = np.concatenate((n_coils * [(0, 1)], n_coils * [(-np.pi, np.pi)]))

    def cost(weights):
        return cov(combine_maps(b1_roi, vector_to_complex(weights)))

    shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, bounds=bounds).x)
    shim_weights = shim_weights / np.linalg.norm(shim_weights)
    print(f"Shim coefficient: {shim_weights}")
    print(f"Coefficient of variation after shimming: {cov(combine_maps(b1_roi, shim_weights))}")
    return shim_weights


def combine_maps(b1_maps, weights):
    """
    Combines the B1 field distribution of several coils into one map representing the total B1 field magnitude.

    Args:
        b1_maps (numpy.ndarray): Measured B1 field for different coils. Last dimension must correspond to n_coils.
        weights (numpy.ndarray): 1D complex array of length n_coils.

    Returns:

    """
    if b1_maps.shape[-1] == len(weights):
        pass
    else:
        raise ValueError("The number of shim weights does not match the number of coils.")

    return abs(np.sum(np.multiply(b1_maps, weights), b1_maps.ndim - 1))


def cov(array):
    """
    Computes the coefficient of variation (CoV) of a given array.

    Args:
        array (numpy.ndarray): N-dimensional array.

    Returns:
        float: Coefficient of variation of the input array (standard deviation/mean).

    """
    return np.std(array) / np.mean(array)


def vector_to_complex(weights):
    """
    Combines magnitude and phase values contained in a vector into a half long complex vector.

    Args:
        weights (numpy.ndarray): 1D array of length 2*n_coils. First/second half: real/imaginary part of shim weights.

    Returns:
        numpy.ndarray: 1D complex array of length n_coils.

    """
    if len(weights) % 2 == 0:
        pass
    else:
        raise ValueError("The vector must have an even number of elements.")
    return weights[:len(weights) // 2] * np.exp(1j * weights[len(weights) // 2:])


def complex_to_vector(weights):
    """
    Combines separates magnitude and phase values contained in a complex vector into a twice as long vector.

    Args:
        weights (numpy.ndarray): 1D complex array of length n_coils.

    Returns:
        numpy.ndarray: 1D array of length 2*n_coils. First/second half: real/imaginary part of shim weights.

    """
    return np.concatenate((np.abs(weights), np.angle(weights)))
