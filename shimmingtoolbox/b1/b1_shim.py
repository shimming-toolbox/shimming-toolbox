#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import scipy.optimize
import logging
import warnings


def b1_shim(b1_maps, mask, cp_weights=None, vop=None, SED=1.5, constrained=False):
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
        b1_roi = np.reshape(b1_maps * mask[:, :, :, np.newaxis], [x * y * n_slices, n_coils])
        b1_roi = b1_roi[b1_roi[:, 0] != 0, :]
    else:
        raise ValueError("Mask and maps dimensions not matching.")

    if cp_weights:
        if len(cp_weights) == b1_maps.shape[-1]:
            if np.isclose(np.linalg.norm(cp_weights), 1, rtol=0.0001):
                weights_init = complex_to_vector(cp_weights)
            else:
                warnings.warn("Normalizing the CP mode weights.")
                weights_init = complex_to_vector(cp_weights / np.linalg.norm(cp_weights))
        else:
            raise ValueError("CP mode and maps dimensions not matching.")
    else:
        weights_init = complex_to_vector(calc_cp(b1_maps))

    # Bounds for the optimization
    bounds = np.concatenate((n_coils * [(0, None)], n_coils * [(-np.pi, np.pi)]))

    # # SAR constraint
    # sar_limit = SED * np.max(np.real(vector_to_complex(weights_init) @ vop.T @ vector_to_complex(weights_init)).T)
    # cons = ({'type': 'ineq', 'fun': lambda weights: np.max(np.real(np.matmul(vector_to_complex(weights), np.matmul(vop.T, vector_to_complex(
    #                                                                                    weights)).T)))})

    def cost(weights):
        return cov(combine_maps(b1_roi, vector_to_complex(weights)))

    shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, bounds=bounds).x)
    shim_weights = shim_weights / np.linalg.norm(shim_weights)

    return shim_weights


def combine_maps(b1_maps, weights):
    """
    Combines the B1 field distribution of several coils into one map representing the total B1 field magnitude.

    Args:
        b1_maps (numpy.ndarray): Complex B1 field for different coils (x, y, n_slices, n_coils).
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
    return array.std() / array.mean()


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


def calc_cp(b1_maps, voxel_position=None, voxel_size=None):
    """
    Reads in B1 maps and returns the individual shim weights for each channel that correspond to a circular polarization
    (CP) mode, computed in the specified voxel.
    Args:
        b1_maps (numpy.ndarray): Complex B1 field for different coils (x, y, n_slices, n_coils).
        voxel_position (numpy.ndarray): Position of the center of the voxel considered to compute the CP mode
        voxel_size (numpy.ndarray): Size of the voxel

    Returns:
        numpy.ndarray: Complex 1D array of individual shim weights (length = n_coils).

    """
    x, y, z, n_coils = b1_maps.shape
    if voxel_size is None:
        voxel_size = np.asarray([5, 5, 1])  # Default voxel size
        warnings.warn("No voxel size provided for CP computation. Default size set to a single voxel.")
    else:
        if (voxel_size > np.asarray([x, y, z])).any():
            raise ValueError("The size of the voxel used for CP computation exceeds the size of the B1 maps.")

    if voxel_position is None:
        voxel_position = (x//2, y//2, z//2)
        warnings.warn("No voxel position provided for CP computation. Default set to the center of the B1 maps.")
    else:
        if (voxel_position < np.asarray([0, 0, 0])).any() or (voxel_position > np.asarray([x, y, z])).any():
            raise ValueError("The position of the voxel used to compute the CP mode exceeds the B1 maps bounds.")

    start_voxel = voxel_position - voxel_size // 2
    end_voxel = start_voxel + voxel_size

    if (start_voxel < 0).any() or (end_voxel > np.asarray([x, y, z])).any():
        raise ValueError("Voxel bounds exceed the B1 maps.")

    cp_phases = np.zeros(n_coils, dtype=complex)
    for channel in range(n_coils):
        voxel_values = b1_maps[start_voxel[0]:end_voxel[0], start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2], channel]
        mean_phase = np.angle(voxel_values).mean()

        if channel == 0:
            mean_phase_first_channel = mean_phase

        cp_phases[channel] = -1*mean_phase - mean_phase_first_channel

    cp_weights = (np.ones(n_coils)*np.exp(1j*cp_phases))/np.linalg.norm(np.ones(n_coils))

    return cp_weights

