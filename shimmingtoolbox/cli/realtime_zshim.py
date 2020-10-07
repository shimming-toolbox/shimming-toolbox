#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(
    context_settings=CONTEXT_SETTINGS,
    help=f"Perform realtime z-shimming."
)
@click.option("-coil", help="Coil basis to use for shimming. Enter multiple files if you wish to use more than one set "
                            "of shim coils (eg: Siemens gradient/shim coils and external custom coils).")
@click.option("-fmap", help="B0 fieldmap. For realtime shimming, this should be a 4d file (4th dimension being time")
@click.option("-mask", help="3D nifti file with voxels between 0 and 1 used to weight the spatial region to shim.")
@click.option("-verbose", is_flag=True, help="Be more verbose.")
def realtime_zshim(coil, fmap, mask, verbose):

    nii_coil = nib.load(coil)
    nii_fmap = nib.load(fmap)
    # TODO: check good practice below
    if mask is not None:
        nii_mask = nib.load(mask)
    else:
        nii_mask = None


    currents = np.zeros([8, nt])
    shimmed = np.zeros_like(fieldmaps)
    masked_fieldmaps = np.zeros_like(fieldmaps)
    masked_shimmed = np.zeros_like(shimmed)
    for i_t in range(nt):
        currents[:, i_t] = sequential_zslice(fieldmaps[:, :, :, i_t], basis, full_mask, z_slices=np.array(range(nz)))
        shimmed[:, :, :, i_t] = fieldmaps[:, :, :, i_t] + np.sum(currents[:, i_t] * basis, axis=3, keepdims=False)
        masked_fieldmaps[:, :, :, i_t] = full_mask * fieldmaps[:, :, :, i_t]
        masked_shimmed[:, :, :, i_t] = full_mask * shimmed[:, :, :, i_t]

    i_t = 0
    # Plot results
    fig = Figure(figsize=(10, 10))
    # FigureCanvas(fig)
    ax = fig.add_subplot(2, 2, 1)
    im = ax.imshow(masked_fieldmaps[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Masked unshimmed")
    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(masked_shimmed[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Masked shimmed")
    ax = fig.add_subplot(2, 2, 3)
    im = ax.imshow(fieldmaps[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Unshimmed")
    ax = fig.add_subplot(2, 2, 4)
    im = ax.imshow(shimmed[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Shimmed")

    print(f"\nThe associated current coefficients are : {currents[:, i_t]}")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_plot.png')
    fig.savefig(fname_figure)
    return fname_figure


if __name__ == '__main__':
    realtime_zshim()
