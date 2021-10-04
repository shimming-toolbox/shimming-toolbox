#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import scipy.optimize
import logging
from scipy.stats import variation as cov

logger = logging.getLogger(__name__)


def b1_shim(b1_maps, mask, cp_weights=None, algo=1, reg_param=0, target=None,  q_matrix=None, SED=1.5):
    """
    Computes static optimized shim weights that minimize the B1 field coefficient of variation over the masked region.

    Args:
        b1_maps (numpy.ndarray): 4D array  corresponding to the measured B1 field. (x, y, n_slices, n_channels)
        mask (numpy.ndarray): 3D array corresponding to the region where shimming will be performed. (x, y, n_slices)
        cp_weights (numpy.ndarray): 1D vector of length n_channels of complex weights corresponding to the CP mode of
        the coil. Must be normalized.
        algo (int): Number from 1 to 5 specifying which algorithm to use for B1 optimization:
                    1 - Optimization aiming to reduce the coefficient of variation (CoV) of the resulting B1+ field.
                    2 - Magnitude least square (MLS) optimization targeting a specific B1+ value. Target value required.
                    3 - Maximizes the minimum B1+ value for better efficiency.
        reg_param (float): RF power Tikhonov-regularization parameter used by algorithms 2 and 3. Default is 0.
        target (float): Target B1+ value used by algorithm 2 in nT/V. Required by algorithm 2.
        q_matrix (numpy.ndarray): Matrix used to constrain local SAR. If no matrix is provided, unconstrained
        optimization is performed, which might result in SAR excess at the scanner (n_channels, n_channels, n_vop).
        SED (float): Factor (=> 1) to which the local SAR after optimization can exceed the CP mode local SAR. SED
        between 1 and 1.5 usually work with Siemens scanners

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

    if cp_weights is not None:
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

    if algo == 1:
        # CoV minimization
        def cost(weights):
            return cov(combine_maps(b1_roi, vector_to_complex(weights)))

    elif algo == 2:
        # MLS targeting value
        if target is None:
            raise ValueError(f"Algorithm 2 requires a target B1 value in nT/V.")

        def cost(weights):
            b1_abs = combine_maps(b1_roi, vector_to_complex(weights))
            return np.square(np.linalg.norm(b1_abs - target)) + reg_param * (1 / np.mean(np.square(b1_abs)))

    elif algo == 3:
        # Maximizing the minimum B1+ value to get better RF efficiency
        def cost(weights):
            b1_abs = combine_maps(b1_roi, vector_to_complex(weights))
            return 1 / np.min(b1_abs)

    else:
        raise ValueError(f"The specified algorithm does not exist. It must be an integer between 1 and 5.")

    if q_matrix is not None:
        if SED < 1:
            raise ValueError(f"The SAR factor must be equal to or greater than 1.")
        max_sar = SED * sar(vector_to_complex(weights_init), q_matrix)
        cons = ({'type': 'ineq', 'fun': lambda w: -sar(vector_to_complex(w), q_matrix) + max_sar})  # SAR constraint
        shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, constraints=cons).x)
    else:
        logger.info(f"No Q matrix provided, performing unconstrained optimization.")
        shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init).x)

    return shim_weights


def combine_maps(b1_maps, weights):
    """
    Combines the B1 field distribution of several channels into one map representing the total B1 field magnitude.

    Args:
        b1_maps (numpy.ndarray): Complex B1 field for different channels (x, y, n_slices, n_channels).
        weights (numpy.ndarray): 1D complex array of length n_channels.

    Returns:
        numpy.ndarray: B1 field distribution obtained when applying the provided shim weights.
    """
    if b1_maps.shape[-1] == len(weights):
        pass
    else:
        raise ValueError(f"The number of shim weights does not match the number of channels.\n"
                         f"Number of shim weights: {len(weights)}\n"
                         f"Number of channels: {b1_maps.shape[-1]}")

    return np.abs(np.matmul(b1_maps, weights))


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
        b1_maps (numpy.ndarray): Complex B1 field for different channels. (x, y, n_slices, n_channels)
        voxel_position (tuple): Position of the center of the voxel considered to compute the CP mode.
        voxel_size (tuple): Size of the voxel.

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
        voxel_position = (x // 2, y // 2, n_slices // 2)
        logger.info("No voxel position provided for CP computation. Default set to the center of the B1 maps.")
    else:
        if (np.asarray(voxel_position) < np.asarray([0, 0, 0])).any() or \
                (np.asarray(voxel_position) > np.asarray([x, y, n_slices])).any():
            raise ValueError(f"The position of the voxel used to compute the CP mode exceeds the B1 maps bounds.\n"
                             f"B1 maps size: {b1_maps.shape[:-1]}\n"
                             f"Voxel position: {voxel_position}")
    start_voxel = np.asarray(voxel_position) - np.asarray(voxel_size) // 2
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

    cp_weights = (np.ones(n_channels) * np.exp(1j * cp_phases)) / np.linalg.norm(np.ones(n_channels))

    return cp_weights


def calc_approx_cp(n_channels):
    """
    Returns a approximation of a circular polarization based on the number of transmit elements. Assumes a circular coil
    with regularly spaced transmit elements
    Args:
        n_channels (int): Number of transmit elements to consider.

    Returns:
        numpy.ndarray: Complex 1D array of individual shim weights (length: n_channels).

    """

    # Approximation of a circular polarisation
    return (np.ones(n_channels) * np.exp(-1j * np.linspace(0, 2 * np.pi - 2 * np.pi / n_channels, n_channels))) / \
        np.linalg.norm(np.ones(n_channels))


def sar(weights, q_matrix):
    """
    Returns the maximum local SAR corresponding to a set of shim weight and a set of Q matrices
    Args:
        weights (numpy.ndarray): 1D vector of complex shim weights. (length: n_channel)
        q_matrix (numpy.ndarray): Q matrices used to compute the local energy deposition in the tissues.
        (n_channels, n_channels, n_voxel)

    Returns:
        float: maximum local SAR.
    """

    return np.max(np.real(np.matmul(np.matmul(np.conj(weights).T, q_matrix.T), weights)))
