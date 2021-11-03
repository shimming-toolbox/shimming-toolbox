#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.load_nifti import read_nii
from shimmingtoolbox.shim.b1shim import b1shim
from shimmingtoolbox.utils import create_output_dir
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--b1map', 'fname_b1_map', required=True, type=click.Path(exists=True),
              help="Path to B1 nifti as returned by st_dicom_to_nifti.")  # TODO consider using pre scaled B1 maps here
@click.option('--mask', 'fname_mask', type=click.Path(exists=True), required=False,
              help="3D nifti file used to define the static spatial region to shim. "
                   "The coordinate system should be the same as ``b1map``'s coordinate system.")
@click.option('--cp', 'fname_cp_weights', type=click.Path(exists=True), required=False,
              help="json file containing CP weights. See config/cp_mode.json for example.")
@click.option('--algo', 'algorithm', type=int, default=1, show_default=True,
              help="Number from 1 to 3 specifying which algorithm to use for B1 optimization"
                   "1 - Optimization aiming to reduce the coefficient of variation (CoV) of the resulting B1+ field."
                   "2 - Magnitude least square optimization targeting a specific B1+ value. Target value required."
                   "3 - Maximizes the minimum B1+ value for better efficiency.")
@click.option('--target', 'target', type=float, required=False,
              help="Target B1+ value used by algorithm 2 in nT/V")
@click.option('--sed', 'sed', type=float, required=False,
              help="Factor (=> 1) to which the local SAR after optimization can exceed the CP mode local SAR."
                   "SED between 1 and 1.5 usually work with Siemens scanners. Higher SED allows more liberty for RF"
                   "shimming but might result in SAR excess at the scanner.")
@click.option('-o', '--output', 'path_output', type=click.Path(),
              default=os.path.join(os.curdir, 'b1_shim_results'), show_default=True,
              help="Directory to output shim weights text file and figures.")
def b1shim_cli(fname_b1_map, fname_mask, fname_cp_weights=None, algorithm=1, target=None, q_matrix=None, sed=1.5,
                path_output=None):
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

    # Load static anatomical mask
    if fname_mask is not None:
        nii_mask = nib.load(fname_mask)
    else:
        nii_mask = None

    # If a path to a cp json file is provided, read it and store the values as complex numbers
    # See example of json file in config/cp_mode.json (Do not add spaces within the complex values)
    if fname_cp_weights is not None:
        with open(fname_cp_weights) as json_file:
            cp_json = json.load(json_file)
        n_cp_weights = len(cp_json["weights"])
        cp_weights = np.zeros(n_cp_weights, dtype=complex)
        i = 0
        for weight in cp_json["weights"]:
            cp_weights[i] = complex(weight)
            i += 1
    else:
        cp_weights = None

    shim_weights = b1shim(b1_map, mask=nii_mask, cp_weights=cp_weights, algorithm=algorithm, target=target,
                           q_matrix=q_matrix, sed=sed, path_output=path_output)

    # Write to a text file
    fname_output = os.path.join(path_output, 'RF_shim_weights.txt')
    file_rf_shim_weights = open(fname_output, 'w')
    file_rf_shim_weights.write(f'Channel\tmag\tphase (\u00b0)\n')
    for i_channel in range(len(shim_weights)):
        file_rf_shim_weights.write(f'Tx{i_channel + 1}\t{np.abs(shim_weights[i_channel]):.3f}\t'
                                   f'{np.rad2deg(np.angle(shim_weights[i_channel])):.3f}\n')
    file_rf_shim_weights.close()

    return shim_weights
