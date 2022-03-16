#!usr/bin/env python3
# -*- coding: utf-8

import logging
import numpy as np
import os
import scipy.io
import scipy.optimize

from scipy.stats import variation
from shimmingtoolbox.masking.threshold import threshold

logger = logging.getLogger(__name__)


def b1shim(b1, mask=None, algorithm=1, target=None, q_matrix=None, sar_factor=1.5):
    """
    Computes static optimized shim weights that minimize the B1+ field coefficient of variation over the masked region.

    Args:
        b1 (numpy.ndarray): 4D array  corresponding to the measured B1+ field. (x, y, n_slices, n_channels)
        mask (numpy.ndarray): 3D array corresponding to the region where shimming will be performed. (x, y, n_slices)
        algorithm (int): Number from 1 to 4 specifying which algorithm to use for B1+ optimization:
                    1 - Optimization aiming to reduce the coefficient of variation (CV) of the resulting B1+ field.
                    2 - Magnitude least square (MLS) optimization targeting a specific B1+ value. Target value required.
                    3 - Maximizes the SAR efficiency (B1+/sqrt(SAR)). Q matrices required.
                    4 - Phase-only shimming.
        target (float): Target B1+ value used by algorithm 2 in nT/V.
        q_matrix (numpy.ndarray): Matrix used to constrain local SAR. If no matrix is provided, unconstrained
            optimization is performed, which might result in SAR excess at the scanner (n_channels, n_channels, n_vop).
        sar_factor (float): Factor (=> 1) to which the maximum local SAR after shimming can exceed the phase-only
            shimming maximum local SAR. Values between 1 and 1.5 should work with Siemens scanners. High factors allow
            more shimming liberty but are more likely to result in SAR excess at the scanner.

    Returns:
        numpy.ndarray: Optimized and normalized 1D vector of complex shimming weights of length n_channels.
    """
    if b1.ndim == 4:
        x, y, n_slices, n_channels = b1.shape
    else:
        raise ValueError(f"The provided B1 maps have an unexpected number of dimensions.\nExpected: 4\nActual: "
                         f"{b1.ndim}")

    if mask is None:
        # If no mask provided, mask = 1 for every pixel where b1_maps values are non-zero
        logger.info("No mask provided, masking all zero-valued pixels.")
        mask = threshold(b1.sum(axis=-1), thr=0)

    if b1.shape[:-1] == mask.shape:
        # b1_roi will be a (n_pixels, n_channels) numpy array with all zero-valued pixel removed
        b1_roi = np.reshape(b1 * mask[:, :, :, np.newaxis], [x * y * n_slices, n_channels])
        b1_roi = b1_roi[b1_roi.sum(axis=-1) != 0, :]  # Remove all zero values from the ROI
    else:
        raise ValueError(f"Mask and maps dimensions not matching.\n"
                         f"Maps dimensions: {b1.shape[:-1]}\n"
                         f"Mask dimensions: {mask.shape}")

    if not b1_roi.any():
        raise ValueError("The mask does not overlap with the B1+ values.")

    # Phase-only optimization starting from zero phase values
    weights_phase_only = phase_only_shimming(b1_roi)
    # Perform 3 additional iterations with random initial phases and keep the one that yields the best homogeneity
    for i in range(3):
        weights_phase_only_tmp = phase_only_shimming(b1_roi, init_phases=2*np.pi*np.random.rand(n_channels))
        if variation(combine_maps(b1_roi, weights_phase_only_tmp)) < variation(combine_maps(b1_roi, weights_phase_only)):
            weights_phase_only = weights_phase_only_tmp

    algorithm = int(algorithm)
    if algorithm == 4:
        # Phase-only shimming
        return weights_phase_only  # If phase-only shimming is selected, stop the execution here

    # For complex B1+ shimming, use the phase only shimming weights as a starting point for optimization
    # The complex shim weights must be reshaped as a real vector during the optimization
    weights_init = complex_to_vector(weights_phase_only)

    # Initialize empty constraint for optimization
    constraint = []

    if algorithm == 1:
        # CV minimization
        def cost(weights):
            b1_abs = combine_maps(b1_roi, vector_to_complex(weights))
            # Regularize with mean B1+ efficiency/2000 to favor high efficiency solutions. Avoid convergence towards 0
            # 2000 was determined empirically. It did not affect the obtained CV for any testing data
            return variation(b1_abs) - b1_abs.mean()/2000

    elif algorithm == 2:
        # MLS targeting value
        if target is None:
            raise ValueError("Algorithm 2 requires a target B1 value in nT/V.")

        def cost(weights):
            b1_abs = combine_maps(b1_roi, vector_to_complex(weights))
            return np.sum((b1_abs - target)**2)

    elif algorithm == 3:
        if q_matrix is None:
            raise ValueError("Algorithm 3 requires Q matrices to perform SAR efficiency shimming.")

        # Maximizing the minimum B1+ value to get better RF efficiency
        def cost(weights):
            b1_abs = combine_maps(b1_roi, vector_to_complex(weights))
            # Return inverse of mean SAR efficiency
            return np.sqrt(max_sar(vector_to_complex(weights), q_matrix)) / np.mean(b1_abs)

    else:
        raise ValueError("The specified algorithm does not exist. It must be an integer between 1 and 4.")

    # Q matrices to compute the local SAR values for each 10g of tissue (or subgroups of pixels if VOP are used).
    # If no Q matrix is provided, unconstrained optimization is performed
    if q_matrix is not None:
        if sar_factor < 1:
            raise ValueError(f"The SAR factor must be equal to or greater than 1.")
        sar_limit = sar_factor * max_sar(vector_to_complex(weights_init), q_matrix)
        # Create SAR constraint
        sar_constraint = ({'type': 'ineq', 'fun': lambda w: -max_sar(vector_to_complex(w), q_matrix) + sar_limit})
        constraint.append(sar_constraint)
    else:
        norm_cons = ({'type': 'eq', 'fun': lambda w: np.linalg.norm(vector_to_complex(w)) - 1})  # Norm constraint
        constraint.append(norm_cons)
        logger.info(f"No Q matrix provided, performing SAR unconstrained optimization while keeping the RF shim-weighs "
                    f"normalized.")
    # Optimize the complex shim weights
    shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, constraints=constraint).x)
    # Apply phase shift to set the phase of the first channel to 0
    shim_weights = np.abs(shim_weights) * np.exp(1j * (np.angle(shim_weights) - np.angle(shim_weights[0])))
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
    Combines real and imaginary values contained in a vector into a half long complex vector.

    Args:
        weights (numpy.ndarray): 1D array of length 2*n_channels. First/second half: real/imaginary.

    Returns:
        numpy.ndarray: 1D complex array of length n_channels.

    """
    if len(weights) % 2 == 0:
        pass
    else:
        raise ValueError("The vector must have an even number of elements.")
    return weights[:len(weights) // 2] + 1j * weights[len(weights) // 2:]


def complex_to_vector(weights):
    """
    Separates the real and imaginary components of a complex vector into a twice as long vector.

    Args:
        weights (numpy.ndarray): 1D complex array of length n_channels.

    Returns:
        numpy.ndarray: 1D array of length 2*n_channels. First/second half: real/imaginary.

    """
    return np.concatenate((np.real(weights), np.imag(weights)))


def max_sar(weights, q_matrix):
    """
    Returns the maximum local SAR corresponding to a set of shim weight and a set of Q matrices.

    Args:
        weights (numpy.ndarray): 1D vector of complex shim weights. (length: n_channel)
        q_matrix (numpy.ndarray): Q matrices used to compute the local energy deposition in the tissues.
        (n_channels, n_channels, n_voxel)

    Returns:
        float: maximum local SAR.
    """

    return np.max(np.real(np.matmul(np.matmul(np.conj(weights).T, q_matrix.T), weights)))


def load_siemens_vop(fname_sar_file):
    """
    Reads in a Matlab file in which the VOP matrices are stored and returns them as a numpy array.

    Args:
        fname_sar_file: Path to the 'SarDataUser.mat' file containing the scanner's VOPs. This file should be available
            at the scanner in 'C:/Medcom/MriProduct/PhysConfig'.

    Returns:
        numpy.ndarray: VOP matrices (n_coils, n_coils, n_VOPs)

    """
    # Check that the file exists
    if os.path.exists(fname_sar_file):
        sar_data = scipy.io.loadmat(fname_sar_file)
    else:
        raise FileNotFoundError('The SarDataUser.mat file could not be found.')

    if 'ZZ' not in sar_data:
        raise ValueError('The SAR data does not contain the expected VOP values.')

    return sar_data['ZZ']
    # Only return VOPs corresponding to 6 (body parts) and 8 (allowed forward power by channel):
    # return sar_data['ZZ'][:, :, np.argwhere(np.logical_or(sar_data['ZZtype'] == 6, sar_data['ZZtype'] == 8))[:, 1]]


def phase_only_shimming(b1_maps, init_phases=None):
    """
    Performs a phase-only RF-shimming to find a set of phases that homogenizes the B1+ field.

    Args:
        b1_maps (numpy.ndarray): 4D array corresponding to the measured B1 field. (x, y, n_slices, n_channels)
        init_phases (numpy.ndarray): 1D array of initial phase values used for optimization.
    Returns:
        numpy.ndarray: Optimized and normalized 1D vector of complex shimming weights of length n_channels.
    """
    n_channels = b1_maps.shape[-1]

    # If no initial phases are provided, set them to 0
    if init_phases is None:
        init_phases = np.zeros(n_channels)
    else:
        if len(init_phases) == n_channels:
            pass
        else:
            raise ValueError(f"The number of phase values ({len(init_phases)}) does not match the number of channels ("
                             f"{n_channels}).")

    def cost_function(phases):
        return variation(combine_maps(b1_maps, np.exp(1j * phases)/np.sqrt(n_channels)))

    shimmed_phases = scipy.optimize.minimize(cost_function, init_phases).x
    # Set phase of first Tx channel to 0 (reference)
    shimmed_phases -= shimmed_phases[0]

    return np.exp(1j * shimmed_phases)/np.sqrt(n_channels)
