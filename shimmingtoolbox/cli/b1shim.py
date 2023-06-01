#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from nibabel.processing import resample_from_to
from shimmingtoolbox.shim.b1shim import b1shim, load_siemens_vop, combine_maps
from shimmingtoolbox.utils import create_output_dir, montage
from shimmingtoolbox.masking.threshold import threshold
from scipy.stats import variation

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--b1', 'fname_b1', required=True, type=click.Path(exists=True),
              help="Complex 3D B1+ map.")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True), required=False, help="3D boolean mask.")
@click.option('--algo', 'algorithm', type=click.Choice(['1', '2', '3', '4']), default='1', show_default=True,
              help="""\b
              Number specifying the B1+ shimming algorithm:
              1 - Reduce the coefficient of variation of the B1+ field. Favors high B1+ efficiency.
              2 - Target a specified B1+ value. Target value required.
              3 - Maximize minimum B1+ for higher signal.
              4 - Phase-only shimming.
              """)
@click.option('--target', 'target', type=float, required=False, help="B1+ value (nT/V) targeted by algorithm 2.")
@click.option('--vop', 'fname_vop', type=click.Path(exists=True), required=False,
              help="SarDataUser.mat file containing VOP matrices used for SAR constraint. Found on the scanner in "
                   "C:/Medcom/MriProduct/PhysConfig.")
@click.option('--sar_factor', 'sar_factor', type=float, required=False, default=1.5,
              help="Factor (=> 1) to which the shimmed max local SAR can exceed the phase-only shimming max local SAR."
                   "Values between 1 and 1.5 should work with Siemens scanners. High factors allow more shimming "
                   "liberty but are more likely to result in SAR excess at the scanner.")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.join(os.curdir, 'b1_shim_results'),
              show_default=True, help="Output directory for shim weights, B1+ maps and figures.")
def b1shim_cli(fname_b1, fname_mask, algorithm, target, fname_vop, sar_factor, path_output):
    """ Perform static B1+ shimming over the volume defined by the mask. This function will generate a text file
    containing shim weights for each transmit element.
    """

    # Create output folder
    create_output_dir(path_output)

    # Load B1 map
    nii_b1 = nib.load(fname_b1)
    with open(fname_b1.split('.nii')[0] + '.json') as json_b1_file:
        json_b1 = json.load(json_b1_file)
    b1_map = np.array(nii_b1.dataobj)

    # Load static anatomical mask
    if fname_mask is not None:
        nii_mask = nib.load(fname_mask)
        # Recombine the Tx B1+ maps to get same dimensions as the mask for resampling
        nii_b1_map_combined = nib.Nifti1Image(b1_map.sum(axis=-1), nii_b1.affine, header=nii_b1.header)
        mask_resampled = resample_from_to(nii_mask, nii_b1_map_combined).get_fdata()
    else:
        mask_resampled = None

    if fname_vop is not None:
        vop = load_siemens_vop(fname_vop)
    else:
        vop = None

    shim_weights = b1shim(b1_map, mask=mask_resampled, algorithm=algorithm, target=target, q_matrix=vop,
                          sar_factor=sar_factor)

    # Save shimmed combined B1+ map in a NIfTI file that can be opened in FSLeyes
    json_b1["ImageComments"] = 'Shimmed B1+ map (nT/V)'
    fname_nii_b1_shim = os.path.join(path_output, 'TB1map_shimmed.nii.gz')
    nii_b1_shim = nib.Nifti1Image(b1_map @ shim_weights, nii_b1.affine, header=nii_b1.header)
    nib.save(nii_b1_shim, fname_nii_b1_shim)
    file_json_b1_shim = open(os.path.join(path_output, 'TB1map_shimmed.json'), mode='w')
    json.dump(json_b1, file_json_b1_shim)

    # Write to a text file
    fname_output_weights = os.path.join(path_output, 'b1_shim_weights_hrd.txt')
    with open(fname_output_weights, 'w') as file_rf_shim_weights:

        file_rf_shim_weights.write(f'Channel\tmag\tphase (\u00b0)\n')
        for i_channel in range(len(shim_weights)):
            file_rf_shim_weights.write(f'Tx{i_channel + 1}\t{np.abs(shim_weights[i_channel]):.3f}\t'
                                       f'{np.rad2deg(np.angle(shim_weights[i_channel])):.3f}\n')

    fname_output_weights = os.path.join(path_output, 'b1_shim_weights.txt')
    with open(fname_output_weights, 'w') as file_rf_shim_weights:
        for i_channel in range(len(shim_weights)):
            file_rf_shim_weights.write(f"{np.abs(shim_weights[i_channel]):.3f} "
                                       f"{np.rad2deg(np.angle(shim_weights[i_channel])):.3f} ")

    # Plot B1+ shimming results
    b1_shimmed = montage(combine_maps(b1_map, shim_weights))  # B1+ shimming result
    if mask_resampled is not None:
        b1_shimmed_masked = b1_shimmed*montage(mask_resampled)
    else:
        b1_shimmed_masked = b1_shimmed*montage(threshold(b1_map.sum(axis=-1), thr=0))

    b1_shimmed_masked[b1_shimmed_masked == 0] = np.nan  # Replace 0 values by nans for image transparency
    vmax = 2*np.nanmean(b1_shimmed_masked)  # Set the maximum display range value to twice the mean B1+ in ROI

    plt.figure()
    plt.imshow(b1_shimmed, vmax=vmax, cmap='gray')  # Display background in gray
    plt.imshow(b1_shimmed_masked, vmin=0, vmax=vmax, cmap='viridis')  # Overlay colored shimming ROI
    plt.axis('off')
    plt.title(r"$\mathregular{B_1^+}$ field after shimming"
              f"\nMean (ROI): {np.nanmean(b1_shimmed_masked):.3} nT/V\n"
              f"CV (ROI): {variation(b1_shimmed_masked[~np.isnan(b1_shimmed_masked)]):.3f}")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='3.5%', pad=0.05)
    cbar = plt.colorbar(cax=cax)
    cbar.ax.set_title('nT/V', fontsize=12)
    cbar.ax.tick_params(size=0)
    fname_figure = os.path.join(path_output, 'b1_shim_results.png')
    plt.savefig(fname_figure)

    # Indicate output path to the user
    print(f"\nB1+ shimming results located in: file://{os.path.abspath(path_output)}\n")

    return shim_weights
