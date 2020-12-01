#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
import json

from shimmingtoolbox.shim.realtime_zshim import realtime_zshim

from sklearn.linear_model import LinearRegression
from nibabel.processing import resample_from_to
# TODO: remove matplotlib and dirtesting import
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.utils import st_progress_bar
from shimmingtoolbox.coils.coordinates import generate_meshgrid

DEBUG = True
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help=f"Perform realtime z-shimming."
)
@click.option('-fmap', 'fname_fmap', required=True, type=click.Path(),
              help="B0 fieldmap. For realtime shimming, this should be a 4d file (4th dimension being time")
@click.option('-mask', 'fname_mask_anat', type=click.Path(), required=False,
              help="3D nifti file with voxels between 0 and 1 used to weight the spatial region to shim. "
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('-resp', 'fname_resp', type=click.Path(),
              help="Siemens respiratory file containing pressure data.")
@click.option('-anat', 'fname_anat', type=click.Path(),
              help="Filename of the anatomical image to apply the correction.")
@click.option('-output', 'fname_output', type=click.Path(),
              help="Directory to output gradient text file and figures")
# TODO: Remove json file as input
@click.option('-json', 'fname_json', type=click.Path(),
              help="Filename of json corresponding BIDS sidecar.")
@click.option("-verbose", is_flag=True, help="Be more verbose.")
def realtime_zshim_cli(fname_fmap, fname_mask_anat, fname_resp, fname_json, fname_anat, fname_output, verbose=True):
    """

    Args:
        fname_fmap:
        fname_mask_anat:
        fname_resp:
        fname_json:
        fname_anat:
        fname_output
        verbose:

    Returns:

    """

    # Load fieldmap
    nii_fmap = nib.load(fname_fmap)

    # Load anat
    nii_anat = nib.load(fname_anat)

    # Load anatomical mask
    if fname_mask_anat is not None:
        nii_mask_anat = nib.load(fname_mask_anat)
    else:
        nii_mask_anat = None

    # TODO: Add json to fieldmap instead of asking for another json file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    pmu = PmuResp(fname_resp)

    static_correction, riro_correction, mean_p = realtime_zshim(nii_fmap, nii_anat, pmu, json_data,
                                                                nii_mask_anat=nii_mask_anat)

    # Look if output directory exists, if not, create it
    if not os.path.exists(fname_output):
        os.makedirs(fname_output)

    # Write to a text file
    fname_corrections = os.path.join(fname_output, 'zshim_gradients.txt')
    file_gradients = open(fname_corrections, 'w')
    for i_slice in range(static_correction.shape[-1]):
        file_gradients.write(f'Vector_Gz[0][{i_slice}]= {static_correction[i_slice]:.6f}\n')
        file_gradients.write(f'Vector_Gz[1][{i_slice}]= {riro_correction[i_slice]:.12f}\n')
        file_gradients.write(f'Vector_Gz[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    return fname_corrections
