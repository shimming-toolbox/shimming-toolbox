#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import json
import logging
import math
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase
from shimmingtoolbox.prepare_fieldmap import get_mask, correct_2pi_offset, VALIDITY_THRESHOLD
from shimmingtoolbox.utils import create_fname_from_path, set_all_loggers, create_output_dir, save_nii_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FILE_OUTPUT_DEFAULT = 'unwrapped.nii.gz'
MASK_OUTPUT_DEFAULT = 'mask_unwrap.nii.gz'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'fname_data', type=click.Path(exists=True), required=True,
              help="Input path of data nifti file")
@click.option('--mag', 'fname_mag', type=click.Path(exists=True), required=True,
              help="Input path of mag nifti file")
@click.option('--unwrapper', type=click.Choice(['prelude', 'skimage']), default='prelude',
              show_default=True,
              help="Algorithm for unwrapping. skimage is installed by default, prelude requires FSL to be installed.")
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT),
              show_default=True, help="Output filename for the unwrapped data, supported types : '.nii', '.nii.gz'")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True),
              help="Input path for a mask.")
@click.option('--threshold', 'threshold', type=float, show_default=True, default=0.05,
              help="Threshold for masking if no mask is provided. Allowed range: [0, 1] where all scaled values lower "
                   "than the threshold are set to 0.")
@click.option('--savemask', 'fname_save_mask', type=click.Path(),
              help="Filename of the mask calculated by the unwrapper")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
def unwrap_cli(fname_data, fname_mag, unwrapper, fname_output, fname_mask, threshold, fname_save_mask, verbose):
    """
    Unwraps images. This algorithm expects the input to have wraps. Edge cases might occur if no wraps are present.
    The unwrapper tries to correct 2pi ambiguity when unwrapping by bringing the mean closest to 0 in increments of 2pi
    jumps.
    """
    # Set logger level
    set_all_loggers(verbose)

    # Create filename for the output if it's a path
    fname_output_v2 = create_fname_from_path(fname_output, FILE_OUTPUT_DEFAULT)

    # Create output directory if it doesn't exist
    create_output_dir(fname_output_v2, is_file=True)

    # Save mask
    if fname_save_mask is not None:
        # If it is a path, add the default filename and create output directory
        fname_save_mask = create_fname_from_path(fname_save_mask, MASK_OUTPUT_DEFAULT)
        create_output_dir(fname_save_mask, is_file=True)

    # Scale to radians
    nii_data = nib.load(fname_data)
    data = nii_data.get_fdata()

    # Scale the input from -pi to pi
    # If the input is wrapped, the unwrapper will unwrap normally, if not, no unwrapping will be done
    scalar = get_scalar_to_fit_2pi(data)
    data_mean = np.mean(data * scalar)
    data_scaled = (data * scalar) - data_mean  # [-pi, pi]
    nii_scaled = nib.Nifti1Image(data_scaled, nii_data.affine, header=nii_data.header)

    # Load magnitude
    mag = nib.load(fname_mag).get_fdata()

    # Load mask
    if fname_mask is not None:
        nii_mask = nib.load(fname_mask)
    else:
        nii_mask = None

    # Returns a mask. If a mask is provided, use that mask. If not, create a mask using the threshold.
    mask = get_mask(nii_scaled, mag, nii_mask, threshold)

    # Unwrap
    unwrapped_rad = unwrap_phase(nii_scaled, unwrapper, mag, mask, fname_save_mask=fname_save_mask)

    # Correct for 2pi offset between timepoints if in 4d, bring closest to the mean if in 3D
    unwrapped_rad = correct_2pi_offset(unwrapped_rad, mag, mask, VALIDITY_THRESHOLD)

    # Scale back to original range
    unwrapped = (unwrapped_rad + data_mean) / scalar
    nii_unwrapped = nib.Nifti1Image(unwrapped.astype(nii_data.get_data_dtype()),
                                    nii_data.affine,
                                    header=nii_data.header)

    # Open and save JSON file
    # Save unwrapped nii
    json_path = fname_data.split('.nii')[0] + '.json'
    if os.path.isfile(json_path):
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        # Save NIfTI and json file to their respective filenames
        save_nii_json(nii_unwrapped, json_data, fname_output_v2)
    else:
        # Save NIfTI
        nib.save(nii_unwrapped, fname_output_v2)

    # Log output file
    logger.info(f"Filename of the unwrapped data is located: {fname_output_v2}")


def get_scalar_to_fit_2pi(data):
    """Return the scalar that scales the data to a range of 2pi"""
    # Scale to radians
    extent = np.max(data) - np.min(data)
    scale = 2 * math.pi / extent
    return scale
