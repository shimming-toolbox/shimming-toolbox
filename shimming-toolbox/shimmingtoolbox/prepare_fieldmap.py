#!/usr/bin/python3
# -*- coding: utf-8 -*

import copy
import logging
import math
import nibabel as nib
import numpy as np
from skimage.filters import gaussian
from sklearn.linear_model import LinearRegression

from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.masking.threshold import threshold as mask_threshold
from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase
from shimmingtoolbox.utils import fill

logger = logging.getLogger(__name__)
# Threshold to apply when creating a mask to remove 2*pi offsets
VALIDITY_THRESHOLD = 0.2


def prepare_fieldmap(list_nii_phase, echo_times, mag, unwrapper='prelude', nii_mask=None, threshold=0.05,
                     gaussian_filter=False, sigma=1, fname_save_mask=None):
    """ Creates fieldmap (in Hz) from phase images. This function accommodates multiple echoes (2 or more) and phase
    difference. This function also accommodates 4D phase inputs, where the 4th dimension represents the time, in case
    multiple field maps are acquired across time for the purpose of real-time shimming experiments.

    Args:
        list_nii_phase (list): List of nib.Nifti1Image phase values. The array can be [x, y], [x, y, z] or [x, y, z, t].
                               The values must range from [-pi to pi].
        echo_times (list): List of echo times in seconds for each echo. The number of echotimes must match the number of
                           echoes. It input is a phasediff (1 phase), input 2 echotimes.
        unwrapper (str): Unwrapper to use for phase unwrapping. Supported: ``prelude``, ``skimage``.
        mag (numpy.ndarray): Array containing magnitude data relevant for ``phase`` input. Shape must match phase[echo].
        nii_mask (nib.Nifti1Image): Mask for masking output fieldmap.
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
            if (phase[i_echo].max() > math.pi + 1e-3) or (phase[i_echo].min() < -math.pi - 1e-3):
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

    mask = get_mask(list_nii_phase[0], mag, nii_mask=nii_mask, threshold=threshold)

    # Get the time between echoes and calculate phase difference depending on number of echoes
    if len(phase) == 1:
        # phase should be a phasediff
        nii_phasediff = list_nii_phase[0]
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]
        # Run the unwrapper
        phasediff_unwrapped = unwrap_phase(nii_phasediff, unwrapper=unwrapper, mag=mag, mask=mask,
                                           fname_save_mask=fname_save_mask)

        # If 4d, correct 2pi offset between timepoints, if it's 3d, bring offset closest to the mean
        phasediff_unwrapped = correct_2pi_offset(phasediff_unwrapped, mag, mask, VALIDITY_THRESHOLD)

        # Divide by echo time
        fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]

    elif len(phase) == 2:
        echo_0 = phase[0]
        echo_1 = phase[1]

        # Calculate phasediff using complex difference
        phasediff = complex_difference(echo_0, echo_1)
        nii_phasediff = nib.Nifti1Image(phasediff, list_nii_phase[0].affine, header=list_nii_phase[0].header)

        # Calculate the echo time difference
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]
        # Run the unwrapper
        phasediff_unwrapped = unwrap_phase(nii_phasediff, unwrapper=unwrapper, mag=mag, mask=mask,
                                           fname_save_mask=fname_save_mask)

        # If it's 4d (i.e. there are timepoints)
        if len(phasediff_unwrapped.shape) == 4:
            phasediff_unwrapped = correct_2pi_offset(phasediff_unwrapped, mag, mask, VALIDITY_THRESHOLD)

        # Divide by echo time
        fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]

    else:
        # Calculates field map based on multi echo phases by running the unwrapper for each echo individually.
        n_echoes = len(list_nii_phase)  # Number of Echoes
        unwrapped = [unwrap_phase(list_nii_phase[echo_number], unwrapper=unwrapper, mag=mag, mask=mask,
                                  fname_save_mask=fname_save_mask) for echo_number in range(n_echoes)]
        unwrapped_data = np.moveaxis(np.stack(unwrapped, axis=0), 0, -1)  # Merges all phase echoes on the last dim
        # The mag must be the same size as the unwrapped_data to yield an equal mask for the
        # "correct_2pi_offset" function.
        new_mag = np.repeat(mag[..., np.newaxis], n_echoes, axis=-1)
        mask_tmp = np.repeat(mask[..., np.newaxis], n_echoes, axis=-1)

        # Time series
        if len(np.shape(list_nii_phase[0])) == 4:
            # dimensions: [x, y, z, t, echo]
            unwrapped_data_corrected = np.zeros_like(unwrapped_data)
            # Correct all time points individually
            n_t = np.shape(list_nii_phase[0])[3]
            for i_t in range(n_t):
                unwrapped_data_corrected[..., i_t, :] = correct_2pi_offset(unwrapped_data[..., i_t, :],
                                                                           new_mag[..., i_t, :], mask_tmp[..., i_t, :],
                                                                           VALIDITY_THRESHOLD)
        # One time point
        else:
            # dimensions: [x, y, z, echo]
            unwrapped_data_corrected = correct_2pi_offset(unwrapped_data, new_mag, mask_tmp, VALIDITY_THRESHOLD)

        x = np.asarray(echo_times)
        # Calculates multi linear regression for the whole "unwrapped_data_corrected" as Y and "echo_times" as X.
        # So, X and Y reshaped into [n_echoes * 1] array and [n_echoes * total number of voxels / phase] respectively.
        reg = LinearRegression().fit(x.reshape(-1, 1), unwrapped_data_corrected.reshape(-1, n_echoes).T)
        # Slope of linear regression reshaped into the shape of original 3D phase.
        fieldmap_rad = reg.coef_.reshape(unwrapped_data.shape[:-1])  # [rad / s]

    fieldmap_hz = fieldmap_rad / (2 * math.pi)  # [Hz]

    # Gaussian blur the fieldmap
    if gaussian_filter:
        # If its 4d data, gaussian blur each volume individually
        if len(fieldmap_hz.shape) == 4:
            for it in range(fieldmap_hz.shape[-1]):
                # Fill values
                filled = fill(fieldmap_hz[..., it], mask[..., it] == False)
                fieldmap_hz[..., it] = gaussian(filled, sigma, mode='nearest') * mask[..., it]
        # 3d data
        else:
            filled = fill(fieldmap_hz, mask == False)
            fieldmap_hz = gaussian(filled, sigma, mode='nearest') * mask

    return fieldmap_hz, mask


def get_mask(nii_target, mag, nii_mask=None, threshold=None):
    """ Return a mask resampled (if required) to nii_target. If nii_mask is None, a mask is created using the threshold.
        This functions hanles 3D and 4D nii_targets.

    Args:
        nii_target (nib.Nifti1Image): Target nifti image to resample the mask to.
        nii_mask (nib.Nifti1Image): Mask to be resampled to nii_target. If None, a mask is created using the threshold.
        mag (np.ndarray): Magnitude data to create the mask from.
        threshold (float): Threshold to create the mask. If nii_mask is not None, this value is ignored.

    Returns:
        np.ndarray: Mask resampled to nii_target.
    """
    # Make sure mask has the right shape
    if nii_mask is None:
        # Define the mask using the threshold
        mask = mask_threshold(mag, threshold, scaled_thr=True)
    else:
        # Check that the mask is the right shape
        if not np.all(nii_mask.shape == nii_target.shape) or not np.all(
                nii_mask.affine == nii_target.affine):
            logger.debug("Resampling mask on the target anat")
            if nii_target.ndim == 4:
                nii_tmp_target = nib.Nifti1Image(nii_target.get_fdata()[..., 0], nii_target.affine,
                                                 header=nii_target.header)
                if nii_mask.ndim == 3:
                    tmp_mask = np.repeat(nii_mask.get_fdata()[..., np.newaxis], nii_target.shape[-1], axis=-1)
                    nii_tmp_mask = nib.Nifti1Image(tmp_mask, nii_mask.affine, header=nii_mask.header)
                elif nii_mask.ndim == 4:
                    nii_tmp_mask = nii_mask
                else:
                    raise ValueError("Mask must be 3D or 4D")
            else:
                # If it's not in 4d, assume it's a 3d mask
                nii_tmp_target = nii_target
                nii_tmp_mask = nii_mask

            nii_mask_soft = resample_from_to(nii_tmp_mask, nii_tmp_target, order=1, mode='grid-constant')
            tmp_mask = nii_mask_soft.get_fdata()
            # Change soft mask into binary mask
            mask = mask_threshold(tmp_mask, thr=0.001, scaled_thr=True)
        else:
            mask = nii_mask.get_fdata()

        logger.debug("A mask was provided, ignoring threshold value")

    return mask


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
    """ Removes 2*pi offsets from `unwrapped` for a time series. If there is no offset, it returns the same array. The
        'correct' offset is assumed to be at time 0.

    Args:
        unwrapped (numpy.ndarray): Array of the spatially unwrapped phase. If there is a time dimension, the offset is
                                   corrected in time, if unwrapped is 3D, the offset closest to 0 is chosen.
        mag (numpy.ndarray): Array containing the magnitude values of the phase. Same shape as unwrapped.
        mask (numpy.ndarray): Mask of the unwrapped phase array. Same shape as unwrapped.
        validity_threshold (float): Threshold to create a mask on each timepoints and assume as reliable phase data

    Returns:
        numpy.ndarray: 4d array of the unwrapped phase corrected if there were n*2*pi offsets between time points

    """
    unwrapped_cp = copy.deepcopy(unwrapped)
    # Create a mask that excludes the noise
    # TODO: What if the validity region is bigger than the mask
    validity_masks = mask_threshold(mag - mag.min(), validity_threshold * (mag.max() - mag.min()))

    # Logical and with the mask used for calculating the fieldmap
    validity_masks = np.logical_and(mask, validity_masks)

    if unwrapped.ndim == 4:
        for i_time in range(0, unwrapped_cp.shape[3]):
            # Correct the first time point and bring closest to the mean
            if i_time == 0:
                unwrapped_cp[..., i_time] = correct_2pi_offset(unwrapped_cp[..., i_time],
                                                               mag[..., i_time],
                                                               mask[..., i_time],
                                                               validity_threshold)
                continue

            # Take the region where both masks intersect
            validity_mask = np.logical_and(validity_masks[..., i_time - 1], validity_masks[..., i_time])

            # Calculate the means in the same validity mask
            ma_0 = np.ma.array(unwrapped_cp[..., i_time - 1], mask=validity_mask == False)
            mean_0 = np.ma.mean(ma_0)
            ma_1 = np.ma.array(unwrapped_cp[..., i_time], mask=validity_mask == False)
            mean_1 = np.ma.mean(ma_1)

            # Calculate the number of offset by rounding to the nearest integer.
            n_offsets_float = (mean_1 - mean_0) / (2 * np.pi)
            n_offsets = np.round(n_offsets_float)

            if 0.3 < (n_offsets_float % 1) < 0.7:
                logger.warning("The number of 2*pi offsets when calculating the fieldmap is close to "
                               "ambiguous, verify the output fieldmap.")

            if n_offsets != 0:
                logger.info(f"Correcting for n*2pi phase offset, 'n' was: {n_offsets_float}")

            logger.debug(f"Offset was: {n_offsets_float}")
            # Remove n_offsets to unwrapped[..., i_time] only in the masked region
            unwrapped_cp[..., i_time] -= mask[..., i_time] * n_offsets * (2 * np.pi)
    else:
        # unwrapped.ndim == 3:
        ma_0 = np.ma.array(unwrapped_cp, mask=validity_masks == False)
        mean_0 = np.ma.mean(ma_0)
        n_offsets_float = mean_0 / (2 * np.pi)
        n_offsets = np.round(n_offsets_float)
        if n_offsets != 0:
            logger.info(f"Correcting for n*2pi phase offset, 'n' was: {n_offsets_float}")

        unwrapped_cp -= mask * n_offsets * (2 * np.pi)

    return unwrapped_cp
