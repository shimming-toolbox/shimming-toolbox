#!usr/bin/env python3
# -*- coding: utf-8

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.optimize

from scipy.stats import variation as cov
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.utils import montage

logger = logging.getLogger(__name__)


def b1shim(b1_maps, path_output, mask=None, algorithm=1, target=None,  q_matrix=None, sed=1.5):
    """
    Computes static optimized shim weights that minimize the B1 field coefficient of variation over the masked region.

    Args:
        b1_maps (numpy.ndarray): 4D array  corresponding to the measured B1 field. (x, y, n_slices, n_channels)
        path_output (str): Path to output figures and RF shim weights.
        mask (numpy.ndarray): 3D array corresponding to the region where shimming will be performed. (x, y, n_slices)
        algorithm (int): Number from 1 to 3 specifying which algorithm to use for B1 optimization:
                    1 - Optimization aiming to reduce the coefficient of variation (CoV) of the resulting B1+ field.
                    2 - Magnitude least square (MLS) optimization targeting a specific B1+ value. Target value required.
                    3 - Maximizes the SAR efficiency (B1+/sqrt(SAR)). Q matrices required.
        target (float): Target B1+ value used by algorithm 2 in nT/V.
        q_matrix (numpy.ndarray): Matrix used to constrain local SAR. If no matrix is provided, unconstrained
            optimization is performed, which might result in SAR excess at the scanner (n_channels, n_channels, n_vop).
        sed (float): Factor (=> 1) to which the maximum local SAR after shimming can exceed the phase-only shimming
            maximum local SAR. SED between 1 and 1.5 usually work with Siemens scanners. Higher SED gives more shimming
            freedom but might result in SAR excess at the scanner.

    Returns:
        numpy.ndarray: Optimized and normalized 1D vector of complex shimming weights of length n_channels.
    """
    if b1_maps.ndim == 4:
        x, y, n_slices, n_channels = b1_maps.shape
    else:
        raise ValueError(f"The provided B1 maps have an unexpected number of dimensions.\nExpected: 4\nActual: "
                         f"{b1_maps.ndim}")

    if mask is None:
        # If no mask provided, mask = 1 for every pixel where b1_maps values are non-zero
        logger.info("No mask provided, masking all zero-valued pixels.")
        mask = threshold(b1_maps.sum(axis=-1), thr=0)

    if b1_maps.shape[:-1] == mask.shape:
        # b1_roi will be a (n_pixels, n_channels) numpy array with all zero-valued pixel removed
        b1_roi = np.reshape(b1_maps * mask[:, :, :, np.newaxis], [x * y * n_slices, n_channels])
        b1_roi = b1_roi[b1_roi.sum(axis=-1) != 0, :]  # Remove all zero values from the ROI
    else:
        raise ValueError(f"Mask and maps dimensions not matching.\n"
                         f"Maps dimensions: {b1_maps.shape[:-1]}\n"
                         f"Mask dimensions: {mask.shape}")

    # Initial weights for optimization are obtained by performing a phase-only shimming prior to the RF-shimming
    weights_phase_only = phase_only_shimming(b1_roi)
    # The complex shim weights must be reshaped as a real vector during the optimization
    weights_init = complex_to_vector(weights_phase_only)

    algorithm = int(algorithm)
    if algorithm == 1:
        # CoV minimization
        def cost(weights):
            return cov(combine_maps(b1_roi, vector_to_complex(weights)))

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
        raise ValueError("The specified algorithm does not exist. It must be an integer between 1 and 3.")

    # Q matrices to compute the local SAR values for each 10g of tissue (or subgroups of pixels if VOP are used).
    # If no Q matrix is provided, unconstrained optimization is performed
    if q_matrix is not None:
        if sed < 1:
            raise ValueError(f"The SAR factor must be equal to or greater than 1.")
        sar_limit = sed * max_sar(vector_to_complex(weights_init), q_matrix)
        # Create SAR constraint
        sar_constraint = ({'type': 'ineq', 'fun': lambda w: -max_sar(vector_to_complex(w), q_matrix) + sar_limit})
        shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, constraints=sar_constraint).x)
    else:
        norm_cons = ({'type': 'eq', 'fun': lambda w: np.linalg.norm(vector_to_complex(w)) - 1})  # Norm constraint
        logger.info(f"No Q matrix provided, performing SAR unconstrained optimization while keeping the RF shim-weighs "
                    f"normalized.")
        shim_weights = vector_to_complex(scipy.optimize.minimize(cost, weights_init, constraints=norm_cons).x)

    # Plot RF shimming results
    b1_phase_only = montage(combine_maps(b1_maps, weights_phase_only))  # Phase-only shimming result
    b1_phase_only_masked = b1_phase_only*montage(mask)
    b1_phase_only_masked[b1_phase_only_masked == 0] = np.nan  # Replace 0 values by nans for image transparency
    b1_shimmed = montage(combine_maps(b1_maps, shim_weights))  # RF-shimming result
    b1_shimmed_masked = b1_shimmed*montage(mask)
    b1_shimmed_masked[b1_shimmed_masked == 0] = np.nan  # Replace 0 values by nans for image transparency
    vmax = np.percentile(np.concatenate((b1_phase_only, b1_shimmed)), 99)  # Reduce high values influence on display
    vmax = 5*np.ceil(vmax/5)  # Ceil max range value to next multiple of 5 for good colorbar display

    fig, ax = plt.subplots(1, 2)
    plt.tight_layout(pad=0)
    ax[0].imshow(b1_phase_only, vmax=vmax, cmap='gray')
    im = ax[0].imshow(b1_phase_only_masked, vmin=0, vmax=vmax, cmap="jet")
    ax[0].axis('off')
    ax[0].set_title(f"$B_1^+$ field (phase-only shimming)\nMean $B_1^+$ in ROI: "
                    f"{np.nanmean(b1_phase_only_masked):.3} nT/V\nCoV in ROI: "
                    f"{cov(b1_phase_only_masked[~np.isnan(b1_phase_only_masked)]):.3}")

    ax[1].imshow(b1_shimmed, vmax=vmax, cmap='gray')
    ax[1].imshow(b1_shimmed_masked, vmin=0, vmax=vmax, cmap="jet")
    ax[1].axis('off')
    ax[1].set_title(f"$B_1^+$ field (RF shimming)\nMean $B_1^+$ in ROI: {np.nanmean(b1_shimmed_masked):.3} nT/V\n"
                    f"CoV in ROI: {cov(b1_shimmed_masked[~np.isnan(b1_shimmed_masked)]):.3f}")

    cax = fig.add_axes([ax[0].get_position().x0, ax[0].get_position().y0 - 0.025,
                        ax[1].get_position().x1-ax[0].get_position().x0, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.set_title('nT/V', fontsize=12, y=-4)
    cbar.ax.tick_params(size=0)
    fname_figure = os.path.join(path_output, 'b1_shim_results.png')
    fig.savefig(fname_figure)

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


def load_siemens_vop(path_sar_file):
    """
    Reads in a Matlab file in which the VOP matrices are stored and returns them as a numpy array.

    Args:
        path_sar_file: Path to the 'SarDataUser.mat' file containing the scanner's VOPs. This file should be available
            at the scanner in 'C:/Medcom/MriProduct/PhysConfig'.

    Returns:
        numpy.ndarray: VOP matrices (n_coils, n_coils, n_VOPs)

    """
    # Check that the file exists
    if os.path.exists(path_sar_file):
        sar_data = scipy.io.loadmat(path_sar_file)
    else:
        raise FileNotFoundError('The SarDataUser.mat file could not be found.')

    if 'ZZ' not in sar_data:
        raise ValueError('The SAR data does not contain the expected VOP values.')

    return sar_data['ZZ']
    # Only return VOPs corresponding to 6 (body parts) and 8 (allowed forward power by channel):
    # return sar_data['ZZ'][:, :, np.argwhere(np.logical_or(sar_data['ZZtype'] == 6, sar_data['ZZtype'] == 8))[:, 1]]


def phase_only_shimming(b1_maps):
    """
    Performs a phase-only RF-shimming to find a set of phases that homogenizes the B1+ field.

    Args:
        b1_maps (numpy.ndarray): 4D array  corresponding to the measured B1 field. (x, y, n_slices, n_channels)

    Returns:
        numpy.ndarray: Optimized and normalized 1D vector of complex shimming weights of length n_channels.
    """
    n_channels = b1_maps.shape[-1]
    # Start phase optimization from null phase values on each channel
    phases_init = np.zeros(n_channels)

    def cost_function(phases):
        return cov(combine_maps(b1_maps, np.exp(1j * phases)/np.sqrt(n_channels)))

    shimmed_phases = scipy.optimize.minimize(cost_function, phases_init).x

    return np.exp(1j * shimmed_phases)/np.sqrt(n_channels)
