#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import scipy.optimize
import logging

logger = logging.getLogger(__name__)


def b1_shim(b1_maps, mask, cp_weights=None, vop=None, SED=1.5, constrained=False):
    """
    Computes static optimized shim weights that minimize the B1 field coefficient of variation over the masked region.

    Args:
        b1_maps (numpy.ndarray): 4D array (x, y, n_slices, n_channels) corresponding to the measured B1 field.
        mask (numpy.ndarray): 3D array (x, y, n_slices) corresponding to the region where shimming will be performed.
        cp_weights (numpy.ndarray): 1D vector of length n_channels of complex weights corresponding to the CP mode of
        the coil. Must be normalized.
        vop (numpy.ndarray): (n_channels, n_channels, n_vop) matrix used to constrain local SAR.
        SED (int): factor to which the local SAR after optimization can exceed the CP mode local SAR.
        constrained (boolean): Specifies if the optimization has to be constrained (True) or unconstrained (False).

    Returns:
        numpy.ndarray: Optimized and normalized 1D vector of complex shimming weights of length n_channels.
    """
    if b1_maps.ndim == 4:
        x, y, n_slices, n_channels = b1_maps.shape
    else:
        raise ValueError(f"The provided B1 maps have an unexpected number of dimensions.\nExpected: 4\nActual: "
                         f"{b1_maps.ndim}")

    if b1_maps.shape[:-1] == mask.shape:
        b1_roi = np.reshape(b1_maps * mask[:, :, :, np.newaxis], [x * y * n_slices, n_channels])
        b1_roi = b1_roi[b1_roi[:, 0] != 0, :]
    else:
        raise ValueError(f"Mask and maps dimensions not matching.\n"
                         f"Maps dimensions: {b1_maps.shape[:-1]}\n"
                         f"Mask dimensions: {mask.shape}")

    if cp_weights:
        if len(cp_weights) == b1_maps.shape[-1]:
            if np.isclose(np.linalg.norm(cp_weights), 1, rtol=0.0001):
                weights_init = complex_to_vector(cp_weights)
            else:
                logger.info("Normalizing the CP mode weights.")
                weights_init = complex_to_vector(cp_weights / np.linalg.norm(cp_weights))
        else:
            raise ValueError(f"The number of CP weights does not match the number of channels.\n"
                             f"Number of CP weights: {len(cp_weights)}\n"
                             f"Number of channels: {b1_maps.shape[-1]}")
    else:
        weights_init = complex_to_vector(calc_cp(b1_maps))

    # Bounds for the optimization
    bounds = np.concatenate((n_channels * [(0, None)], n_channels * [(-np.pi, np.pi)]))

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
    Combines the B1 field distribution of several channels into one map representing the total B1 field magnitude.

    Args:
        b1_maps (numpy.ndarray): Complex B1 field for different channels (x, y, n_slices, n_channels).
        weights (numpy.ndarray): 1D complex array of length n_channels.

    Returns:

    """
    if b1_maps.shape[-1] == len(weights):
        pass
    else:
        raise ValueError(f"The number of shim weights does not match the number of channels.\n"
                         f"Number of shim weights: {len(weights)}\n"
                         f"Number of channels: {b1_maps.shape[-1]}")

    return np.abs(b1_maps @ weights)


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
        weights (numpy.ndarray): 1D array of shim weights (length 2*n_channels). First/second half: magnitude/phase.

    Returns:
        numpy.ndarray: 1D complex array of length n_channels.

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
        weights (numpy.ndarray): 1D complex array of length n_channels.

    Returns:
        numpy.ndarray: 1D array of shim weights (length 2*n_channels). First/second half: magnitude/phase.

    """
    return np.concatenate((np.abs(weights), np.angle(weights)))


def calc_cp(b1_maps, voxel_position=None, voxel_size=None):
    """
    Reads in B1 maps and returns the individual shim weights for each channel that correspond to a circular polarization
    (CP) mode, computed in the specified voxel.
    Args:
        b1_maps (numpy.ndarray): Complex B1 field for different channels (x, y, n_slices, n_channels).
        voxel_position (numpy.ndarray): Position of the center of the voxel considered to compute the CP mode
        voxel_size (tuple): Size of the voxel

    Returns:
        numpy.ndarray: Complex 1D array of individual shim weights (length = n_channels).

    """
    x, y, n_slices, n_channels = b1_maps.shape
    if voxel_size is None:
        if (b1_maps.shape[:-1] < np.asarray([5, 5, 1])).any():
            raise ValueError(f"Provided B1 maps are too small to compute CP phases.\n"
                             f"Minimum size: (5, 5, 1)\n"
                             f"Actual size: {b1_maps.shape[:-1]}")
        voxel_size = (5, 5, 1)  # Default voxel size
        logger.info("No voxel size provided for CP computation. Default size set to (5, 5, 1).")
    else:
        if (np.asarray(voxel_size) > np.asarray([x, y, n_slices])).any():
            raise ValueError(f"The size of the voxel used for CP computation exceeds the size of the B1 maps.\n"
                             f"B1 maps size: {b1_maps.shape[:-1]}\n"
                             f"Voxel size: {voxel_size}")

    if voxel_position is None:
        voxel_position = (x//2, y//2, n_slices//2)
        logger.info("No voxel position provided for CP computation. Default set to the center of the B1 maps.")
    else:
        if (np.asarray(voxel_position) < np.asarray([0, 0, 0])).any() or \
                (np.asarray(voxel_position) > np.asarray([x, y, n_slices])).any():
            raise ValueError(f"The position of the voxel used to compute the CP mode exceeds the B1 maps bounds.\n"
                             f"B1 maps size: {b1_maps.shape[:-1]}\n"
                             f"Voxel position: {voxel_position}")
    start_voxel = np.asarray(voxel_position) - np.asarray(voxel_size)//2
    end_voxel = start_voxel + np.asarray(voxel_size)

    if (start_voxel < 0).any() or (end_voxel > np.asarray([x, y, n_slices])).any():
        raise ValueError("Voxel bounds exceed the B1 maps.")

    cp_phases = np.zeros(n_channels, dtype=complex)
    mean_phase_first_channel = 0
    for channel in range(n_channels):
        values = b1_maps[start_voxel[0]:end_voxel[0], start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2], channel]
        mean_phase = np.angle(values).mean()

        if channel == 0:
            # Remember mean phase of 1st channel as it is used to compute the CP phases of the next channels
            mean_phase_first_channel = mean_phase

        cp_phases[channel] = -(mean_phase - mean_phase_first_channel)

    cp_weights = (np.ones(n_channels)*np.exp(1j*cp_phases))/np.linalg.norm(np.ones(n_channels))

    return cp_weights
