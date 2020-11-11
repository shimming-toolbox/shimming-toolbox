#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os

from shimmingtoolbox.load_nifti import read_nii


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('phase', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-mag', 'fname_mag', type=click.Path(exists=True), required=True, help="Input path of mag nifti file")
@click.option('-output', 'path_output', type=click.Path(), default=os.curdir, help="Output path for the fieldmap")
def prepare_fieldmap_cli(phase, fname_mag, path_output):
    """Creates fieldmap from phase and magnitute images

    Args:
        phase: Input path of phase nifti file"
    """

    # flag: -unwrapper DEFAULT prelude, -method DEFAULT phase diff, -mask DEFAULT threshold
    # Make sure we have a mag input, make sure we have a phase input
    # Look for method, compare to number of inputs (look in header to make sure #inputs work with given input)
    # for phase (phasediff or echoes)

    if len(phase) == 1:
        # phase should be a phasediff
        fname_phasediff = phase[0]
        # TODO: replace by read_nii once bug #170 is merged
        # nii_phasediff, json_phasediff, phasediff = read_nii(fname_phasediff)
        # echo_time_diff =
    elif len(phase) == 2:
        fname_phase0 = phase[0]
        fname_phase1 = phase[1]
    else:
        raise RuntimeError(" Number of phase filename not supported")

    return path_output

