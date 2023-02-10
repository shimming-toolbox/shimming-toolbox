#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os
import nibabel as nib
import json
import logging

from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.prepare_fieldmap import prepare_fieldmap
from shimmingtoolbox.utils import create_fname_from_path, set_all_loggers, create_output_dir, save_nii_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FILE_OUTPUT_DEFAULT = 'fieldmap.nii.gz'
MASK_OUTPUT_DEFAULT = 'mask_fieldmap.nii.gz'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('phase', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--mag', 'fname_mag', type=click.Path(exists=True), required=True, help="Input path of mag nifti file")
@click.option('--unwrapper', type=click.Choice(['prelude']), default='prelude', show_default=True,
              help="Algorithm for unwrapping")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT),
              show_default=True, help="Output filename for the fieldmap, supported types : '.nii', '.nii.gz'")
@click.option('--autoscale-phase', 'autoscale', type=click.BOOL, default=True, show_default=True,
              help="Tells whether to auto rescale phase inputs according to manufacturer standards. If you have non "
                   "standard data, it would be preferable to set this option to False and input your phase data from "
                   "-pi to pi to avoid unwanted rescaling")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True),
              help="Input path for a mask. Mask must be the same shape as the array of each PHASE input.")
@click.option('--threshold', 'threshold', type=float, show_default=True, default=0.05,
              help="Threshold for masking if no mask is provided. Allowed range: [0, 1] where all scaled values lower "
                   "than the threshold are set to 0.")
@click.option('--savemask', 'fname_save_mask', type=click.Path(),
              help="Filename of the mask calculated by the unwrapper")
@click.option('--gaussian-filter', 'gaussian_filter', type=bool, show_default=True, help="Gaussian filter for B0 map")
@click.option('--sigma', type=float, default=1, help="Standard deviation of gaussian filter. Used for: gaussian_filter")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def prepare_fieldmap_cli(phase, fname_mag, unwrapper, fname_output, autoscale, fname_mask, threshold, fname_save_mask,
                         gaussian_filter, sigma, verbose):
    """Creates fieldmap (in Hz) from phase images.

    This function accommodates multiple echoes (2 or more) and phase difference. This function also
    accommodates 4D phase inputs, where the 4th dimension represents the time, in case multiple
    field maps are acquired across time for the purpose of real-time shimming experiments.
    For non Siemens phase data, see --autoscale-phase option.

    PHASE: Input path of phase nifti file(s), in ascending order: echo1, echo2, etc.
    """
    # Set logger level
    set_all_loggers(verbose)

    prepare_fieldmap_uncli(phase, fname_mag, unwrapper, fname_output, autoscale, fname_mask, threshold, fname_save_mask,
                           gaussian_filter, sigma)


def prepare_fieldmap_uncli(phase, fname_mag, unwrapper='prelude',
                           fname_output=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT), autoscale=True,
                           fname_mask=None, threshold=0.05, fname_save_mask=None, gaussian_filter=False, sigma=1):
    """ Prepare fieldmap cli without the click decorators. This allows this function to be imported and called from
    other python modules.

    Args:
        phase (list): Input path of phase nifti file(s), in ascending order: echo1, echo2, etc.
        fname_mag (str): Input path of mag nifti file
        unwrapper (str): Algorithm for unwrapping. Supported unwrapper: 'prelude'.
        fname_output (str): Output filename for the fieldmap, supported types : '.nii', '.nii.gz'
        autoscale (bool): Tells whether to auto rescale phase inputs according to manufacturer standards. If you have
                          non siemens data not automatically converted from dcm2niix, you should set this to False and
                          input phase data from -pi to pi.
        fname_mask (str): Input path for a mask. Mask must be the same shape as the array of each PHASE input.
                          Used for PRELUDE
        threshold (float): Threshold for masking. Used for: PRELUDE
        fname_save_mask (str): Filename of the mask calculated by the unwrapper
        gaussian_filter (bool): Gaussian filter for B0 map
        sigma (float): Standard deviation of gaussian filter. Used for: gaussian_filter
    """
    # Return fieldmap and json file
    nii_fieldmap, json_fieldmap = prepare_fieldmap_cli_inputs(phase, fname_mag, unwrapper, autoscale, fname_mask,
                                                              threshold, fname_save_mask, gaussian_filter, sigma)

    # Create filename for the output if it,s a path
    fname_output_v2 = create_fname_from_path(fname_output, FILE_OUTPUT_DEFAULT)

    # Save fieldmap and json file to their respective filenames
    save_nii_json(nii_fieldmap, json_fieldmap, fname_output_v2)

    # Log output file
    logger.info(f"Filename of the fieldmap is: {fname_output_v2}")


def prepare_fieldmap_cli_inputs(phase, fname_mag, unwrapper, autoscale, fname_mask, threshold, fname_save_mask,
                                gaussian_filter, sigma):
    """Prepare fieldmap using click inputs

    Args:
        phase (list): Input path of phase nifti file(s), in ascending order: echo1, echo2, etc.
        fname_mag (str): Input path of mag nifti file
        unwrapper (str): Algorithm for unwrapping. Supported unwrapper: 'prelude'.
        autoscale (bool): Tells whether to auto rescale phase inputs according to manufacturer standards. If you have
                          non siemens data not automatically converted from dcm2niix, you should set this to False and
                          input phase data from -pi to pi.
        fname_mask (str): Input path for a mask. Mask must be the same shape as the array of each PHASE input.
                          Used for PRELUDE
        threshold (float): Threshold for masking. Used for: PRELUDE
        gaussian_filter (bool): Gaussian filter for B0 map
        sigma (float): Standard deviation of gaussian filter. Used for: gaussian_filter

    Returns:
        (tuple): tuple containing:

            * nib.Nifti1Image: Nibabel object containing the fieldmap in hz.
            * dict: Dictionary containing the json sidecar associated with the nibabel object fieldmap.
    """

    # Save mask
    if fname_save_mask is not None:
        # If it is a path, add the default filename and create output directory
        fname_save_mask = create_fname_from_path(fname_save_mask, MASK_OUTPUT_DEFAULT)

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

    # Magnitude image
    _, json_mag, mag = read_nii(fname_mag, auto_scale=False)

    # Import mask
    if fname_mask is not None:
        mask = nib.load(fname_mask).get_fdata()
    else:
        mask = None

    fieldmap_hz, save_mask = prepare_fieldmap(list_nii_phase, echo_times, mag=mag, unwrapper=unwrapper,
                                              mask=mask, threshold=threshold, gaussian_filter=gaussian_filter,
                                              sigma=sigma, fname_save_mask=fname_save_mask)

    # Save fieldmap
    nii_fieldmap = nib.Nifti1Image(fieldmap_hz, affine, header=nii_phase.header)

    # Save fieldmap json
    json_fieldmap = json_phase
    if len(phase) > 1:
        for i_echo in range(len(echo_times)):
            json_fieldmap[f'EchoTime{i_echo + 1}'] = echo_times[i_echo]
    # save mask json
    if fname_save_mask is not None:
        fname_mask_json = fname_save_mask.rsplit('.nii', 1)[0] + '.json'
        with open(fname_mask_json, 'w') as outfile:
            json.dump(json_mag, outfile, indent=2)

    return nii_fieldmap, json_fieldmap
