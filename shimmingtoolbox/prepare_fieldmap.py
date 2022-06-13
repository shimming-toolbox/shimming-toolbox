#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import math
import nibabel
import numpy as np
from skimage.filters import gaussian

from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase
from shimmingtoolbox.masking.threshold import threshold as mask_threshold

logger = logging.getLogger(__name__)
# Threshold to apply when creating a mask to remove 2*pi offsets
VALIDITY_THRESHOLD = 0.2


def prepare_fieldmap(list_nii_phase, echo_times, mag, unwrapper='prelude', mask=None, threshold=0.05,
                     gaussian_filter=False, sigma=1, fname_save_mask=None):
    """ Creates fieldmap (in Hz) from phase images. This function accommodates multiple echoes (2 or more) and phase
    difference. This function also accommodates 4D phase inputs, where the 4th dimension represents the time, in case
    multiple field maps are acquired across time for the purpose of real-time shimming experiments.

    Args:
        list_nii_phase (list): List of nib.Nifti1Image phase values. The array can be [x, y], [x, y, z] or [x, y, z, t].
                               The values must range from [-pi to pi].
        echo_times (list): List of echo times in seconds for each echo. The number of echotimes must match the number of
                           echoes. It input is a phasediff (1 phase), input 2 echotimes.
        unwrapper (str): Unwrapper to use for phase unwrapping. Supported: prelude.
        mag (numpy.ndarray): Array containing magnitude data relevant for ``phase`` input. Shape must match phase[echo].
        mask (numpy.ndarray): Mask for masking output fieldmap. Must match shape of phase[echo].
        threshold: Threshold for masking if no mask is provided. Allowed range: [0, 1] where all scaled values lower
                   than the threshold are set to 0.
        gaussian_filter (bool): Option of using a Gaussian filter to smooth the fieldmaps (boolean)
        sigma (float): Standard deviation of gaussian filter.
        fname_save_mask (str): Filename of the mask calculated by the unwrapper

    Returns
        numpy.ndarray: Unwrapped fieldmap in Hz.
    """

    phase = [nii_phase.get_fdata() for nii_phase in list_nii_phase]

    # Check inputs
    for i_echo in range(len(phase)):
        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if (phase[i_echo].max() > math.pi) or (phase[i_echo].min() < -math.pi):

            # If this is a rounding error from saving niftis, let it go, the algorithm can handle the difference.
            if (phase[i_echo].max() > math.pi + 1e-6) or (phase[i_echo].min() < -math.pi - 1e-6):
                raise ValueError("Values must range from -pi to pi.")
            else:
                pass
                # phase[i_echo][phase[i_echo] > math.pi] = math.pi
                # phase[i_echo][phase[i_echo] < -math.pi] = -math.pi

    # Check that the input echotimes are the appropriate size by looking at phase
    is_phasediff = (len(phase) == 1 and len(echo_times) == 2)
    if not is_phasediff:
        if len(phase) != len(echo_times) or (len(phase) == 1 and len(echo_times) == 1):
            raise ValueError("The number of echoes must match the number of echo times unless there is 1 echo, which "
                             "requires 2 echo_times")

    # Make sure mag is the right shape
    if mag is not None:
        if mag.shape != phase[0].shape:
            raise ValueError("mag and phase must have the same dimensions.")

    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold should range from 0 to 1. Input value was: {threshold}")

    # Make sure mask has the right shape
    if mask is None:
        # Define the mask using the threshold
        mask = mask_threshold(mag - mag.min(), threshold * (mag.max() - mag.min()))
    else:
        if mask.shape != phase[0].shape:
            raise ValueError("Shape of mask and phase must match.")

        logger.info("A mask was provided, ignoring threshold value")

    # Get the time between echoes and calculate phase difference depending on number of echoes
    if len(phase) == 1:
        # phase should be a phasediff
        nii_phasediff = list_nii_phase[0]
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]

    elif len(phase) == 2:
        echo_0 = phase[0]
        echo_1 = phase[1]

        # Calculate phasediff using complex difference
        phasediff = complex_difference(echo_0, echo_1)
        nii_phasediff = nibabel.Nifti1Image(phasediff, list_nii_phase[0].affine, header=list_nii_phase[0].header)

        # Calculate the echo time difference
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]

    else:
        # TODO: More echoes
        # TODO: Add method once multiple methods are implemented
        raise NotImplementedError(f"This number of phase input is not supported: {len(phase)}.")

    # Run the unwrapper
    phasediff_unwrapped = unwrap_phase(nii_phasediff, unwrapper=unwrapper, mag=mag, mask=mask,
                                       fname_save_mask=fname_save_mask)

    # If it's 4d (i.e. there are timepoints)
    if len(phasediff_unwrapped.shape) == 4:
        phasediff_unwrapped = correct_2pi_offset(phasediff_unwrapped, mag, mask, VALIDITY_THRESHOLD)

    # Divide by echo time
    fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]
    fieldmap_hz = fieldmap_rad / (2 * math.pi)  # [Hz]

    # Gaussian blur the fieldmap
    if gaussian_filter:
        fieldmap_hz = gaussian(fieldmap_hz, sigma, mode='nearest') * mask

    # return fieldmap_hz_gaussian
    return fieldmap_hz, mask


def complex_difference(phase1, phase2):
    """ Calculates the complex difference between 2 phase arrays (phase2 - phase1)

    Args:
        phase1 (numpy.ndarray): Array containing phase data in radians
        phase2 (numpy.ndarray): Array containing phase data in radians. Must be the same shape as phase1.

    Returns:
        numpy.ndarray: The difference in phase between each voxels of phase2 and phase1 (phase2 - phase1)
    """

    # Calculate phasediff using complex difference
    comp_0 = np.ones_like(phase1) * np.exp(-1j * phase1)
    comp_1 = np.ones_like(phase2) * np.exp(1j * phase2)
    return np.angle(comp_0 * comp_1)


def correct_2pi_offset(unwrapped, mag, mask, validity_threshold):
    """ Removes 2*pi offsets from `unwrapped` for a time series. If there is no offset, it returns the same array.

    Args:
        unwrapped (numpy.ndarray): 4d array of the spatially unwrapped phase
        mag (numpy.ndarray): 4d array containing the magnitude values of the phase
        mask (numpy.ndarray): 4d mask of the unwrapped phase array
        validity_threshold (float): Threshold to create a mask on each timepoints and assume as reliable phase data

    Returns:
        numpy.ndarray: 4d array of the unwrapped phase corrected if there were n*2*pi offsets between time points

    """
    # Create a mask that excludes the noise
    validity_masks = mask_threshold(mag - mag.min(), validity_threshold * (mag.max() - mag.min()))

    for i_time in range(1, unwrapped.shape[3]):
        # Take the region where both masks intersect
        validity_mask = np.logical_and(validity_masks[..., i_time - 1], validity_masks[..., i_time])

        # Calculate the means in the same validity mask
        ma_0 = np.ma.array(unwrapped[..., i_time - 1], mask=validity_mask == False)
        mean_0 = np.ma.mean(ma_0)
        ma_1 = np.ma.array(unwrapped[..., i_time], mask=validity_mask == False)
        mean_1 = np.ma.mean(ma_1)

        # Calculate the number of offset by rounding to the nearest integer.
        n_offsets_float = (mean_1 - mean_0) / (2 * np.pi)
        n_offsets = round(n_offsets_float)

        if 0.2 < (n_offsets_float % 1) < 0.8:
            logger.warning("The number of 2*pi offsets when calculating the fieldmap of timepoints is close to "
                           "ambiguous, verify the output fieldmap.")

        if n_offsets != 0:
            logger.debug(f"Correcting for phase n*2pi offset, offset was: {n_offsets_float}")

        # Remove n_offsets to unwrapped[..., i_time] only in the masked region
        unwrapped[..., i_time] -= mask[..., i_time] * n_offsets * (2 * np.pi)

    return unwrapped
