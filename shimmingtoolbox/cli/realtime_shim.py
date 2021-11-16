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
              help="B0 fieldmap in Hertz. This should be a 4d file (4th dimension being time")
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
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True,
              help="Directory to output gradient text file and figures.")
def realtime_shim_cli(fname_fmap, fname_mask_anat_static, fname_mask_anat_riro, fname_resp, fname_anat, fname_output):
    """ Perform realtime xyz-shimming. This function will generate text files containing static and dynamic (due to
    respiration) Gx, Gy, Gz components based on a fieldmap time series and respiratory trace information obtained from
    Siemens bellows (PMUresp_signal.resp). An additional multi-gradient echo (MGRE) magnitude image is used to
    resample the static and dynamic Gx, Gy, Gz component maps to match the MGRE image. Lastly the mean Gx, Gy, Gz
    values within the ROI are computed for each slice. The mean pressure is also generated in the text file to be used
    to shim.
    """

    # Load fieldmap
    nii_fmap = nib.load(fname_fmap)

    # Load anat
    nii_anat = nib.load(fname_anat)
    dim_info = nii_anat.header.get_dim_info()
    if dim_info[2] != 2:
        # Slice must be the 3rd dimension of the file
        # TODO: Reorient nifti so that the slice is the 3rd dim
        raise RuntimeError("Slice encode direction must be the 3rd dimension of the NIfTI file.")

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

    # Reorient according to freq_encode, phase_encode and slice encode direction
    # static
    corr_static_vox = (static_xcorrection, static_ycorrection, static_zcorrection)
    freq_static_corr, phase_static_corr, slice_static_corr = [corr_static_vox[dim] for dim in dim_info]
    # Riro
    corr_riro_vox = (riro_xcorrection, riro_ycorrection, riro_zcorrection)
    freq_riro_corr, phase_riro_corr, slice_riro_corr = [corr_riro_vox[dim] for dim in dim_info]

    # To output to the gradient coord system, axes need some inversions. The gradient coordinate system is defined by
    # the frequency, phase and slice encode directions.
    # TODO: More thorough tests
    phase_encode_is_positive = _get_phase_encode_direction_sign(fname_anat)
    if not phase_encode_is_positive:
        freq_static_corr = -freq_static_corr
        phase_static_corr = -phase_static_corr
        freq_riro_corr = -freq_riro_corr
        phase_riro_corr = -phase_riro_corr

    # Write to a text file
    fname_zcorrections = os.path.join(fname_output, 'zshim_gradients.txt')
    file_gradients = open(fname_zcorrections, 'w')
    for i_slice in range(slice_static_corr.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {slice_static_corr[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {slice_riro_corr[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    fname_ycorrections = os.path.join(fname_output, 'yshim_gradients.txt')
    file_gradients = open(fname_ycorrections, 'w')
    for i_slice in range(phase_static_corr.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {phase_static_corr[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {phase_riro_corr[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    fname_xcorrections = os.path.join(fname_output, 'xshim_gradients.txt')
    file_gradients = open(fname_xcorrections, 'w')
    for i_slice in range(freq_static_corr.shape[-1]):
        file_gradients.write(f'corr_vec[0][{i_slice}]= {freq_static_corr[i_slice]:.6f}\n')
        file_gradients.write(f'corr_vec[1][{i_slice}]= {freq_riro_corr[i_slice] / pressure_rms:.12f}\n')
        file_gradients.write(f'corr_vec[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    return fname_xcorrections, fname_ycorrections, fname_zcorrections


def _get_phase_encode_direction_sign(fname_nii):
    """ Returns the phase encode direction sign

    Args:
        fname_nii (str): Filename to a NIfTI file with its corresponding json file.

    Returns:
        bool: Returns whether the encoding direction is positive (True) or negative (False)
    """

    # Load nibabel
    nii = nib.load(fname_nii)
    dim_info = nii.header.get_dim_info()

    # Load json
    fname_json = fname_nii.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # json_data['PhaseEncodingDirection'] contains i, j or k then a '-' if the direction is reversed
    phase_en_dir = json_data['PhaseEncodingDirection']

    # Check that dim_info is consistent with PhaseEncodingDirection tag i --> 0, j --> 1, k --> 2
    if (phase_en_dir[0] == 'i' and dim_info[1] != 0) \
            or (phase_en_dir[0] == 'j' and dim_info[1] != 1) \
            or (phase_en_dir[0] == 'k' and dim_info[1] != 2):
        raise RuntimeError("Inconsistency between dim_info of fieldmap and PhaseEncodeDirection tag in the json")

    # Find if the phase encode direction is negative or positive
    if len(phase_en_dir) == 2 and phase_en_dir[1] == '-':
        en_is_positive = False
    elif len(phase_en_dir) == 1:
        en_is_positive = True
    else:
        raise ValueError(f"Unexpected value for PhaseEncodingDirection: {phase_en_dir}")

    return en_is_positive
