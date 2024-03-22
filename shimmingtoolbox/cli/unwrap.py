#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
from cloup import command, option, option_group
from cloup.constraints import RequireAtLeast, mutually_exclusive, all_or_none, constraint
import json
import logging
import math
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.prepare_fieldmap import get_mask, correct_2pi_offset, VALIDITY_THRESHOLD
from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase
from shimmingtoolbox.utils import create_fname_from_path, set_all_loggers, create_output_dir, save_nii_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

H_GYROMAGNETIC_RATIO = 42.577478518e+6  # [Hz/T]
ALLOWED_UNITS = ['Hz', 'rad/s', 'T', 'mT', 'G']  # ppm?
FILE_OUTPUT_DEFAULT = 'unwrapped.nii.gz'
MASK_OUTPUT_DEFAULT = 'mask_unwrap.nii.gz'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@command(
    context_settings=CONTEXT_SETTINGS,
)
@option_group("Input/Output",
              option('-i', '--input', 'fname_data', type=click.Path(exists=True), required=True,
                     help="Input path of data nifti file"),
              option('--mag', 'fname_mag', type=click.Path(exists=True), required=True,
                     help="Input path of mag nifti file"),
              option('-o', '--output', 'fname_output', type=click.Path(),
                     default=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT),
                     show_default=True,
                     help="Output filename for the unwrapped data, supported types : '.nii', '.nii.gz'")
              )
@option_group("Scaling options, use either --unit and --dte, or --range.",
              option('--unit', type=click.Choice(ALLOWED_UNITS, case_sensitive=False),
                     required=False,
                     help="Unit of the input data. Used along with --dte to scale the input data."),
              option('--dte', type=click.FLOAT, required=False,
                     help="Delta TE (in seconds). Used along with --unit to scale the input data."),
              option('--range', 'extent', type=click.FLOAT, required=False,
                     help="Range of the input data. Data that can range from [1000, 4095] would have a "
                          "--range of 3095."),
              constraint=RequireAtLeast(1)
              )
@constraint(mutually_exclusive, ['unit', 'extent'])
@constraint(mutually_exclusive, ['dte', 'extent'])
@constraint(all_or_none, ['dte', 'unit'])
@option_group("Mask options",
              option('--mask', 'fname_mask', type=click.Path(exists=True),
                     help="Input path for a mask."),
              option('--threshold', 'threshold', type=click.FLOAT, show_default=True, default=0.05,
                     help="Threshold for masking if no mask is provided. Allowed range: [0, 1] where all scaled values "
                          "lower than the threshold are set to 0."),
              option('--savemask', 'fname_save_mask', type=click.Path(),
                     help="Filename of the mask calculated by the unwrapper")
              )
@option('--unwrapper', type=click.Choice(['prelude', 'skimage']), default='prelude',
        show_default=True,
        help="Algorithm for unwrapping. skimage is installed by default, prelude requires FSL to be installed.")
@option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
        help="Be more verbose")
def unwrap_cli(fname_data, fname_mag, unit, dte, extent, unwrapper, fname_output, fname_mask, threshold,
               fname_save_mask, verbose):
    """
    Unwraps images. This program assumes wraps occur at min() and max() of the data. The unwrapper tries to correct
    2npi ambiguity when unwrapping by bringing the mean closest to 0 in increments of 2pi jumps.
    """
    # Set logger level
    set_all_loggers(verbose)

    # Load data
    nii_data = nib.load(fname_data)
    data = nii_data.get_fdata()

    # Convert 'unit' to upper()
    if unit is not None:
        unit_up = unit.upper()
    else:
        unit_up = None
    units_upper = [a_unit.upper() for a_unit in ALLOWED_UNITS]

    # Scale the input from -pi to pi
    if unit_up in units_upper and dte is not None:
        logger.info(f"Scaling the input data from {unit} to radians using dte: {dte} seconds.")

        if unit_up == units_upper[0]:
            # Hz
            scalar = 2 * math.pi * dte
        elif unit_up == units_upper[1]:
            # rad/s
            scalar = 1 * dte
        elif unit_up == units_upper[2]:
            # T
            scalar = 2 * math.pi * dte * H_GYROMAGNETIC_RATIO
        elif unit_up == units_upper[3]:
            # mT
            scalar = 2 * math.pi * dte * H_GYROMAGNETIC_RATIO * 1e-3
        elif unit_up == units_upper[4]:
            # G
            scalar = 2 * math.pi * dte * H_GYROMAGNETIC_RATIO * 1e-4

    elif extent is not None:
        logger.info(f"Scaling the input data to radians using the --range: {extent}.")
        scalar = _get_scalar_to_fit_2pi(data, extent)

    else:
        raise ValueError("Neither --unit and --dte nor --range were provided."
                         "Use whichever is more convenient for you.")

    data_mean = np.mean(data * scalar)
    data_scaled = (data * scalar) - data_mean  # [-pi, pi]
    nii_scaled = nib.Nifti1Image(data_scaled, nii_data.affine, header=nii_data.header)

    # Create filename for the output if it's a path
    fname_output_v2 = create_fname_from_path(fname_output, FILE_OUTPUT_DEFAULT)

    # Create output directory if it doesn't exist
    create_output_dir(fname_output_v2, is_file=True)

    # Save mask
    if fname_save_mask is not None:
        # If it is a path, add the default filename and create output directory
        fname_save_mask = create_fname_from_path(fname_save_mask, MASK_OUTPUT_DEFAULT)
        create_output_dir(fname_save_mask, is_file=True)

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
    nii_unwrapped = nib.Nifti1Image(unwrapped,
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


def _get_scalar_to_fit_2pi(data, extent=None):
    """Return the scalar that scales the data to a range of 2pi"""

    if extent is None:
        extent = np.max(data) - np.min(data)
        logger.debug(f"--range not provided, using max() - min() to calculate the range: {extent}")
        # A slight offset in the resulting unwrapped image can be introduced if --range is wrong or if it is not
        # provided. Offset at every wrap is: true_range - (max(data) - min(data)) or true_range - provided_range
    else:
        if extent < (np.max(data) - np.min(data)):
            raise ValueError("The provided --range is smaller than the range of the data.")

    # Scale to radians
    scale = 2 * math.pi / extent
    return scale
