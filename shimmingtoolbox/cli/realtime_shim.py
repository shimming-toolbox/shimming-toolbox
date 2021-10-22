#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
import json

from shimmingtoolbox.shim.realtime_shim import realtime_shim
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.utils import create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(),
              help="B0 fieldmap. This should be a 4d file (4th dimension being time")
@click.option('--anat', 'fname_anat', type=click.Path(), required=True,
              help="Filename of the anatomical image to apply the correction.")
@click.option('--resp', 'fname_resp', type=click.Path(), required=True,
              help="Siemens respiratory file containing pressure data.")
@click.option('--mask-static', 'fname_mask_anat_static', type=click.Path(), required=False,
              help="3D nifti file used to define the static spatial region to shim. "
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('--mask-riro', 'fname_mask_anat_riro', type=click.Path(), required=False,
              help="3D nifti file used to define the time varying (i.e. RIRO, Respiration-Induced Resonance Offset) "
                   "spatial region to shim. "
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.curdir,
              help="Directory to output gradient text file and figures.")
def realtime_shim_cli(fname_fmap, fname_mask_anat_static, fname_mask_anat_riro, fname_resp, fname_anat, fname_output):
    """ Perform realtime z-shimming. This function will generate textfile containing static and dynamic (due to
    respiration) Gz components based on a fieldmap time series and respiratory trace information obtained from Siemens
    bellows (PMUresp_signal.resp). An additional multi-gradient echo (MGRE) magnitiude image is used to resample the
    static and dynamic Gz component maps to match the MGRE image. Lastly the mean Gz values within the ROI are computed
    for each slice. The mean pressure is also generated in the text file to be used to shim.
    """

    # Load fieldmap
    nii_fmap = nib.load(fname_fmap)

    # Load anat
    nii_anat = nib.load(fname_anat)

    # Load static anatomical mask
    if fname_mask_anat_static is not None:
        nii_mask_anat_static = nib.load(fname_mask_anat_static)
    else:
        nii_mask_anat_static = None

    # Load riro anatomical mask
    if fname_mask_anat_riro is not None:
        nii_mask_anat_riro = nib.load(fname_mask_anat_riro)
    else:
        nii_mask_anat_riro = None

    fname_json = fname_fmap.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Load PMU
    pmu = PmuResp(fname_resp)

    create_output_dir(fname_output)

    static_xcorrection, static_ycorrection, static_zcorrection, \
        riro_xcorrection, riro_ycorrection, riro_zcorrection, \
        mean_p, pressure_rms = realtime_shim(nii_fmap, nii_anat, pmu, json_data,
                                             nii_mask_anat_static=nii_mask_anat_static,
                                             nii_mask_anat_riro=nii_mask_anat_riro,
                                             path_output=fname_output)

    # Write to a text file
    fname_zcorrections = os.path.join(fname_output, 'zshim_gradients.txt')
    file_gradients = open(fname_zcorrections, 'w')
    for i_slice in range(static_zcorrection.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {static_zcorrection[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {riro_zcorrection[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    fname_ycorrections = os.path.join(fname_output, 'yshim_gradients.txt')
    file_gradients = open(fname_ycorrections, 'w')
    for i_slice in range(static_ycorrection.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {static_ycorrection[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {riro_ycorrection[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    fname_xcorrections = os.path.join(fname_output, 'xshim_gradients.txt')
    file_gradients = open(fname_xcorrections, 'w')
    for i_slice in range(static_xcorrection.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {static_xcorrection[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {riro_xcorrection[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    return fname_xcorrections, fname_ycorrections, fname_zcorrections
