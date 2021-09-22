#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os
import nibabel as nib
import json
import logging

from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.prepare_fieldmap import prepare_fieldmap
from shimmingtoolbox.utils import create_fname_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FILE_OUTPUT_DEFAULT = 'fieldmap.nii.gz'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('phase', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--mag', 'fname_mag', type=click.Path(exists=True), required=False, help="Input path of mag nifti file")
@click.option('--unwrapper', type=click.Choice(['prelude']), default='prelude', show_default=True,
              help="Algorithm for unwrapping")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT),
              show_default=True, help="Output filename for the fieldmap, supported types : '.nii', '.nii.gz'")
@click.option('--autoscale-phase', 'autoscale', type=click.BOOL, default=True, show_default=True,
              help="Tells whether to auto rescale phase inputs according to manufacturer standards. If you have non "
                   "standard data, it would be preferable to set this option to False and input your phase data from "
                   "-pi to pi to avoid unwanted rescaling")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True),
              help="Input path for a mask. Mask must be the same shape as the array of each PHASE input."
                   "Used for PRELUDE")
@click.option('--threshold', 'threshold', type=float, help="Threshold for masking. Used for: PRELUDE")
@click.option('--gaussian-filter', 'gaussian_filter', type=bool, show_default=True, help="Gaussian filter for B0 map")
@click.option('--sigma', type=float, default=1, help="Standard deviation of gaussian filter. Used for: gaussian_filter")
def prepare_fieldmap_cli(phase, fname_mag, unwrapper, fname_output, autoscale, fname_mask, threshold, gaussian_filter,
                         sigma):
    """Creates fieldmap (in Hz) from phase images.

    This function accommodates multiple echoes (2 or more) and phase difference. This function also
    accommodates 4D phase inputs, where the 4th dimension represents the time, in case multiple
    field maps are acquired across time for the purpose of real-time shimming experiments.
    For non Siemens phase data, see --autoscale-phase option.

    PHASE: Input path of phase nifti file(s), in ascending order: echo1, echo2, etc.
    """

    # Make sure output filename is valid
    fname_output_v2 = create_fname_from_path(fname_output, FILE_OUTPUT_DEFAULT)
    if fname_output_v2[-4:] != '.nii' and fname_output_v2[-7:] != '.nii.gz':
        raise ValueError("Output filename must have one of the following extensions: '.nii', '.nii.gz'")

    # Import phase
    list_nii_phase = []
    echo_times = []
    for i_echo in range(len(phase)):
        nii_phase, json_phase, phase_img = read_nii(phase[i_echo], auto_scale=autoscale)

        list_nii_phase.append(nii_phase)
        # Special case for echo_times if input is a phasediff
        if len(phase) == 1:
            # Check that the input phase is indeed a phasediff, by checking the existence of two echo times in the
            # metadata
            if not ('EchoTime1' in json_phase) or not ('EchoTime2' in json_phase):
                raise RuntimeError(
                    "The JSON file of the input phase should include the fields EchoTime1 and EchoTime2 if"
                    "it is a phase difference.")
            echo_times = [json_phase['EchoTime1'], json_phase['EchoTime2']]  # [s]
        else:
            echo_times.append(json_phase['EchoTime'])

    # Get affine from nii
    affine = nii_phase.affine

    # If fname_mag is not an input define mag as None
    if fname_mag is not None:
        mag = nib.load(fname_mag).get_fdata()
    else:
        mag = None

    # Import mask
    if fname_mask is not None:
        mask = nib.load(fname_mask).get_fdata()
    else:
        mask = None

    fieldmap_hz = prepare_fieldmap(list_phase, echo_times, mag=mag, unwrapper=unwrapper,
                                   mask=mask, threshold=threshold, gaussian_filter=gaussian_filter,
                                   sigma=sigma)

    # Save NIFTI
    nii_fieldmap = nib.Nifti1Image(fieldmap_hz, affine, header=nii_phase.header)
    nib.save(nii_fieldmap, fname_output_v2)

    # Save json
    json_fieldmap = json_phase
    if len(phase) > 1:
        for i_echo in range(len(echo_times)):
            json_fieldmap[f'EchoTime{i_echo + 1}'] = echo_times[i_echo]
    fname_json = fname_output_v2.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(json_fieldmap, outfile, indent=2)

    logger.info(f"Filename of the fieldmap is: {fname_output_v2}")
