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
FILE_OUTPUT_DEFAULT = 'TB1map.nii.gz'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('path_rf_nifti', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, FILE_OUTPUT_DEFAULT),
              show_default=True, help="Output filename for the RF map, supported types : '.nii', '.nii.gz'")
def prepare_rfmap_cli(path_rf_nifti, fname_output):
    """Creates complex TX uncombined RF map (in nT/V) from nifti file.

    path_rf_nifti: Input path of rf nifti. The nifti must not have been rescaled yet.
    """

    # Make sure output filename is valid
    fname_output_v2 = create_fname_from_path(fname_output, FILE_OUTPUT_DEFAULT)
    if fname_output_v2[-4:] != '.nii' and fname_output_v2[-7:] != '.nii.gz':
        raise ValueError("Output filename must have one of the following extensions: '.nii', '.nii.gz'")

    info_rfmap, json_rfmap, rfmap = read_nii(path_rf_nifti, auto_scale=True)

    # Get affine from nii
    affine = info_rfmap.affine

    # Save NIFTI
    nii_rfmap= nib.Nifti1Image(rfmap, affine)
    nib.save(nii_rfmap, fname_output)

    # Save json
    fname_json = fname_output.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(json_rfmap, outfile, indent=2)
