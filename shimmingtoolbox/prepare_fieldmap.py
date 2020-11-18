#!/usr/bin/python3
# -*- coding: utf-8 -*

import math
import numpy as np

from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase


def prepare_fieldmap(phase, echo_times, affine, mag=None, unwrapper='prelude', mask=None, threshold=None):
    """ Creates fieldmap from phase and magnitude images

    Args:
        phase (list): List of phase values in a numpy.ndarray. The numpy array can be [x, y], [x, y, z] or [x, y, z, t].
                      The values mustrange from [-pi to pi]
        echo_times (list): List of echo times in seconds for each echo. The number of echotimes must match the number of
                           echos. It inout is a phasediff, (1 phase), input 2 echotimes
        affine (numpy.ndarray): 4x4 affine matrix
        mag (numpy.ndarray): Array containing magnitude data relevant for ``phase`` input. Shape must match phase[echo]
        unwrapper (str): Unwrapper to use for phase unwrapping. Supported: prelude
        mask (numpy.ndarray): Mask for masking output fieldmap. Must match shape of phase[echo]
        threshold: Prelude parameter used for masking.

    Returns
        numpy.ndarray: Unwrapped fieldmap in Hz
    """
    # Check inputs
    for i_echo in range(len(phase)):
        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if (phase[i_echo].max() > math.pi) or (phase[i_echo].min() < -math.pi):
            raise RuntimeError("read_nii must range from -pi to pi")

    # Check that the input phase is indeed a phasediff, by checking the existence of two echo times
    # TODO: maybe there is a more intuitive way to write this test: first check if phasediff, if not: simply assert
    #  len(phase)==len(echo_times)
    if (len(echo_times) != len(phase) and not (len(phase) == 1 and len(echo_times) == 2)) \
            or (len(phase) == 1 and len(echo_times) == 1):
        raise RuntimeError("The number of echoes must match the number of echo times.")

    # If mag is not as an input define it as an array of ones. This is required by 3rd party software such as Prelude.
    # TODO: move this in prelude wrapper
    if mag is not None:
        if mag.shape != phase[0].shape:
            raise RuntimeError("mag and phase must have the same dimensions")
    else:
        mag = np.ones_like(phase[0])

    # Make sure mask has the right shape
    if mask is not None:
        if mask.shape != phase[0].shape:
            raise RuntimeError("Shape of mask and phase must match")

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
        raise NotImplementedError(f"This number of phase input is not supported: {len(phase)}")

    # Run the unwrapper
    phasediff_unwrapped = unwrap_phase(phasediff, mag, affine, unwrapper=unwrapper, mask=mask, threshold=threshold)

    # TODO: correct for potential wraps between time points

    # Divide by echo time
    fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]
    fieldmap_hz = fieldmap_rad / (2 * math.pi)  # [Hz]

    return fieldmap_hz
