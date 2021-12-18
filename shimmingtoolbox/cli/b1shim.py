#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import nibabel as nib
import numpy as np
import os

from nibabel.processing import resample_from_to
from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.shim.b1shim import b1shim, load_siemens_vop
from shimmingtoolbox.utils import create_output_dir
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--b1map', 'fname_b1_map', required=True, type=click.Path(exists=True),
              help="Complex 3D B1+ map.")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True), required=False, help="3D boolean mask.")
@click.option('--algo', 'algorithm', type=click.Choice(['1', '2', '3']), default='1', show_default=True,
              help="""\b
              Number specifying the B1+ shimming algorithm:
                   1 - Reduce B1+ coefficient of variation.
                   2 - Target a specified B1+ value. Target value required.
                   3 - Maximize minimum B1+ for higher signal.
                   """)
@click.option('--target', 'target', type=float, required=False, help="B1+ value (nT/V) targeted by algorithm 2.")
@click.option('--vop', 'fname_vop', type=click.Path(exists=True), required=False,
              help="SarDataUser.mat file containing VOP matrices used for SAR constraint. Found on the scanner in "
                   "C:/Medcom/MriProduct/PhysConfig.")
@click.option('--sed', 'sed', type=float, required=False, default=1.5,
              help="Factor (=> 1) to which the shimmed max local SAR can exceed the phase-only shimming max local SAR."
                   "SED between 1 and 1.5 usually work with Siemens scanners. High SED allows more RF shimming liberty "
                   "but is more likely to result in SAR excess at the scanner.")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.join(os.curdir, 'b1_shim_results'),
              show_default=True, help="Output directory for shim weights, B1+ maps and figures.")
def b1shim_cli(fname_b1_map, fname_mask, algorithm, target, fname_vop, sed, path_output):
    """ Perform static RF shimming over the volume defined by the mask. This function will generate a text file
    containing shim weights for each transmit element.
    """

    # Load B1 map
    nii_b1, json_b1, b1_map = read_nii(fname_b1_map)

    create_output_dir(path_output)

    # Save uncombined B1 map as nifti
    json_b1["ImageComments"] = 'Complex uncombined B1 map (nT/V)'
    fname_nii_b1 = os.path.join(path_output, 'TB1maps_uncombined.nii')
    nib.save(nii_b1, fname_nii_b1)
    file_json_b1 = open(os.path.join(path_output, 'TB1maps_uncombined.json'), mode='w')
    json.dump(json_b1, file_json_b1)
    b1_map_combined = b1_map.sum(axis=-1)
    nii_b1_map_combined = nib.Nifti1Image(b1_map_combined, nii_b1.affine, header=nii_b1.header)

    # Load static anatomical mask
    if fname_mask is not None:
        nii_mask = nib.load(fname_mask)
        mask_resampled = resample_from_to(nii_mask, nii_b1_map_combined, order=1, mode='grid-constant').get_fdata()
    else:
        mask_resampled = None

    if fname_vop is not None:
        vop = load_siemens_vop(fname_vop)
    else:
        vop = None

    shim_weights = b1shim(b1_map, mask=mask_resampled, algorithm=algorithm, target=target,
                          q_matrix=vop, sed=sed, path_output=path_output)

    # Write to a text file
    fname_output = os.path.join(path_output, 'RF_shim_weights.txt')
    file_rf_shim_weights = open(fname_output, 'w')
    file_rf_shim_weights.write(f'Channel\tmag\tphase (\u00b0)\n')
    for i_channel in range(len(shim_weights)):
        file_rf_shim_weights.write(f'Tx{i_channel + 1}\t{np.abs(shim_weights[i_channel]):.3f}\t'
                                   f'{np.rad2deg(np.angle(shim_weights[i_channel])):.3f}\n')
    file_rf_shim_weights.close()

    return shim_weights
