#!/usr/bin/python3
# -*- coding: utf-8 -*

# TODO: implement mask option


import click
import os
import math

from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.unwrap.prelude import prelude

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('phase', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-mag', 'fname_mag', type=click.Path(exists=True), required=False, help="Input path of mag nifti file")
@click.option('-unwrapper', type=click.Choice(['prelude', 'bla']), default='prelude', help="Algorithm for unwrapping")
@click.option('-output', 'path_output', type=click.Path(), default=os.curdir, help="Output path for the fieldmap")
def prepare_fieldmap_cli(phase, fname_mag, unwrapper, path_output):
    """Creates fieldmap from phase and magnitude images

    Args:
        phase: Input path of phase nifti file"
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
        # Check that the input phase is indeed a phasediff, by checking the existence of two echo times in the metadata
        if not ('EchoTime1' in json_phasediff) or not ('EchoTime2' in json_phasediff):
            raise RuntimeError("The JSON file of the input phase should include the fields EchoTime1 and EchoTime2 if"
                               "it is a phase difference.")
        # Check that the output phase is in radian (Note: the test below is not 100% bullet proof)
        if (phasediff.max() >= 2 * math.pi) and (phasediff.min() <= 0):
            raise RuntimeError("read_nii does not support input to convert to radians")
        echo_time_diff = json_phasediff['EchoTime2'] - json_phasediff['EchoTime1']  # [s]

    elif len(phase) == 2:
        fname_phase0 = phase[0]
        fname_phase1 = phase[1]
        # TODO:
        #  - generate phasediff,
        #  - assign variable affine

    else:
        # TODO: More echoes
        raise RuntimeError(" Number of phase filenames not supported")

    if unwrapper == 'prelude':
        # Check mag is input
        # manage 3+ echoes (currently won't work because needs phasediff)
        # TODO: support threshold and mask
        phasediff_unwrapped = prelude(phasediff, mag, affine, mask=None, threshold=None, is_unwrapping_in_2d=False)
    else:
        raise ValueError(f"This option is not available: {unwrapper}")

    # TODO: save NIFTI
