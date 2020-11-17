#!/usr/bin/python3
# -*- coding: utf-8 -*

# TODO: implement mask option


import click
import os
import math
import numpy as np
import nibabel as nib

from nibabel import load as load_nib

from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('phase', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-mag', 'fname_mag', type=click.Path(exists=True), required=False, help="Input path of mag nifti file")
@click.option('-unwrapper', type=click.Choice(['prelude']), default='prelude', help="Algorithm for unwrapping")
@click.option('-output', 'fname_output', type=click.Path(), default=os.curdir, help="Output filename for the fieldmap")
@click.option('-mask', 'fname_mask', type=click.Path(), help="Input path for a mask. Used for PRELUDE")
@click.option('-threshold', 'threshold', type=int, help="Threshold for masking. Used for: PRELUDE")
def prepare_fieldmap_cli(phase, fname_mag, unwrapper, fname_output, fname_mask, threshold):
    """Creates fieldmap from phase and magnitude images

    Args:
        phase: Input path of phase nifti file, oredered in ascending order i.e. echo1, echo2, etc...
    """

    # flag: -unwrapper DEFAULT prelude, -method DEFAULT, mask fname
    # Make sure we have a mag input, make sure we have a phase input
    # Look for method, compare to number of inputs (look in header to make sure #inputs work with given input)
    # for phase (phasediff or echoes)

    # Get the time between echoes and calculate phase difference depending on number of echoes
    if len(phase) == 1:
        # phase should be a phasediff
        fname_phasediff = phase[0]
        nii_phasediff, json_phasediff, phasediff = read_nii(fname_phasediff, auto_scale=True)
        affine = nii_phasediff.affine

        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if (phasediff.max() >= 2 * math.pi) and (phasediff.min() <= 0):
            raise RuntimeError("read_nii does not support input to convert to radians")
        # read_nii returns the phase between 0 and 2pi, prelude requires it to be between -pi and pi so that there is
        # no offset
        phasediff -= math.pi

        # Check that the input phase is indeed a phasediff, by checking the existence of two echo times in the metadata
        if not ('EchoTime1' in json_phasediff) or not ('EchoTime2' in json_phasediff):
            raise RuntimeError("The JSON file of the input phase should include the fields EchoTime1 and EchoTime2 if"
                               "it is a phase difference.")
        echo_time_diff = json_phasediff['EchoTime2'] - json_phasediff['EchoTime1']  # [s]

        # If mag is not as an input define it as an array of ones
        if fname_mag is not None:
            mag = load_nib(fname_mag).get_fdata()
        else:
            mag = np.ones_like(phasediff)

    elif len(phase) == 2:
        # Load niftis
        nii_phasediff_0, json_phasediff_0, phasediff_0 = read_nii(phase[0], auto_scale=True)
        nii_phasediff_1, json_phasediff_1, phasediff_1 = read_nii(phase[1], auto_scale=True)
        affine = nii_phasediff_0.affine
        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if ((phasediff_0.max() >= 2 * math.pi) and (phasediff_0.min() <= 0)) and \
                ((phasediff_1.max() >= 2 * math.pi) and (phasediff_1.min() <= 0)):
            raise RuntimeError("read_nii does not support input to convert to radians")

        # Calculate phasediff using complex difference
        comp_0 = np.ones_like(phasediff_0) * np.exp(-1j * phasediff_0)
        comp_1 = np.ones_like(phasediff_1) * np.exp(1j * phasediff_1)
        phasediff = np.angle(comp_0 * comp_1)

        # Calculate the echo time difference
        echo_time_diff = json_phasediff_1['EchoTime'] - json_phasediff_0['EchoTime']

        # If mag is not as an input define it as an array of ones
        if fname_mag is not None:
            mag = load_nib(fname_mag).get_fdata()
        else:
            mag = np.ones_like(phasediff)

    else:
        # TODO: More echoes
        raise RuntimeError(" Number of phase filenames not supported")

    # Import mask
    if fname_mask is not None:
        mask = nib.load(fname_mask).get_fdata()
        if mask.shape != phasediff.shape:
            raise RuntimeError("Shape of mask and phase must match")
    else:
        mask = None

    # Run the unwrapper
    phasediff_unwrapped = unwrap_phase(phasediff, mag, affine, unwrapper=unwrapper, mask=mask, threshold=threshold)

    # TODO: correct for potential wraps between time points

    # Divide by echo time
    fieldmap_rad = phasediff_unwrapped / echo_time_diff  # [rad / s]
    fieldmap_hz = fieldmap_rad / (2 * math.pi)  # [hz]

    # Save NIFTI
    nii_fieldmap = nib.Nifti1Image(fieldmap_hz, affine)
    nib.save(nii_fieldmap, fname_output)
