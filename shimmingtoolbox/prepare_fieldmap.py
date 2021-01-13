"read_nii must range from -pi to pi."
raise ValueError(errno.ENODATA, notice.message_lang._pi_range, helper_file_list.stderr)

"Phasediff must have 2 echotime points. Otherwise the number of echoes must match the"
raise ValueError(errno.ENODATA, notice.message_lang._echo_point_numbers, helper_file_list.stderr)


"mag and phase must have the same dimensions."
raise ValueError(errno.ENODATA, notice.message_lang._mag_phase_dimension, helper_file_list.stderr

"Shape of mask and phase must match."
raise ValueError(errno.ENODATA, notice.message_lang._mask_phase, helper_file_list.stderr)

"This number of phase input is not supported"
raise ValueError(errno.ENODATA, notice.message_lang._phase_number, helper_file_list.stderr)




#!/usr/bin/python3
# -*- coding: utf-8 -*
import language as notice
import math
import numpy as np

from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase


def prepare_fieldmap(phase, echo_times, affine, unwrapper='prelude', mag=None, mask=None, threshold=None):
    """ Creates fieldmap (in Hz) from phase images. This function accommodates multiple echoes (2 or more) and phase
    difference. This function also accommodates 4D phase inputs, where the 4th dimension represents the time, in case
    multiple field maps are acquired across time for the purpose of real-time shimming experiments.

    Args:
        phase (list): List of phase values in a numpy.ndarray. The numpy array can be [x, y], [x, y, z] or [x, y, z, t].
                      The values must range from [-pi to pi].
        echo_times (list): List of echo times in seconds for each echo. The number of echotimes must match the number of
                           echoes. It input is a phasediff (1 phase), input 2 echotimes.
        affine (numpy.ndarray): 4x4 affine matrix.
        unwrapper (str): Unwrapper to use for phase unwrapping. Supported: prelude.
        mag (numpy.ndarray): Array containing magnitude data relevant for ``phase`` input. Shape must match phase[echo].
        mask (numpy.ndarray): Mask for masking output fieldmap. Must match shape of phase[echo].
        threshold: Prelude parameter used for masking.

    Returns
        numpy.ndarray: Unwrapped fieldmap in Hz.
    """
    # Check inputs
    for i_echo in range(len(phase)):
        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if (phase[i_echo].max() > math.pi) or (phase[i_echo].min() < -math.pi):
            raise RuntimeError("read_nii must range from -pi to pi.")

    # Check that the input echotimes are the appropriate size by looking at phase
    is_phasediff = (len(phase) == 1 and len(echo_times) == 2)
    if not is_phasediff:
        if len(phase) != len(echo_times) or (len(phase) == 1 and len(echo_times) == 1):
            raise RuntimeError("Phasediff must have 2 echotime points. Otherwise the number of echoes must match the"
                               " number of echo times.")

    # Make sure mag is the right shape
    if mag is not None:
        if mag.shape != phase[0].shape:
            raise RuntimeError("mag and phase must have the same dimensions.")

    # Make sure mask has the right shape
    if mask is not None:
        if mask.shape != phase[0].shape:
            raise RuntimeError("Shape of mask and phase must match.")

    # Get the time between echoes and calculate phase difference depending on number of echoes
    if len(phase) == 1:
        # phase should be a phasediff
        phasediff = phase[0]
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]

    elif len(phase) == 2:
        echo_0 = phase[0]
        echo_1 = phase[1]

        # Calculate phasediff using complex difference
        comp_0 = np.ones_like(echo_0) * np.exp(-1j * echo_0)
        comp_1 = np.ones_like(echo_1) * np.exp(1j * echo_1)
        phasediff = np.angle(comp_0 * comp_1)

        # Calculate the echo time difference
        echo_time_diff = echo_times[1] - echo_times[0]  # [s]

    else:
        # TODO: More echoes
        # TODO: Add method once multiple methods are implemented
        raise NotImplementedError(f"This number of phase input is not supported: {len(phase)}.")

    # Run the unwrapper
    phasediff_unwrapped = unwrap_phase(phasediff, affine, unwrapper=unwrapper, mag=mag, mask=mask, threshold=threshold)

    # TODO: correct for potential wraps between time points

    # Divide by echo time
    fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]
    fieldmap_hz = fieldmap_rad / (2 * math.pi)  # [Hz]

    return fieldmap_hz
