#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
# TODO: remove matplotlib import
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox import __dir_shimmingtoolbox__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


# @click.command(
#     context_settings=CONTEXT_SETTINGS,
#     help=f"Perform realtime z-shimming."
# )
# @click.option('-coil', 'fname_coil', required=True, type=click.Path(),
#               help="Coil basis to use for shimming. Enter multiple files if "
#                    "you wish to use more than one set of shim coils (eg: "
#                    "Siemens gradient/shim coils and external custom coils).")
# @click.option('-fmap', 'fname_fmap', required=True, type=click.Path(),
#               help="B0 fieldmap. For realtime shimming, this should be a 4d file (4th dimension being time")
# @click.option('-mask', 'fname_mask', type=click.Path(),
#               help="3D nifti file with voxels between 0 and 1 used to weight the spatial region to shim.")
# @click.option("-verbose", is_flag=True, help="Be more verbose.")
def realtime_zshim(fname_coil, fname_fmap, fname_mask, verbose=True):
    """

    Args:
        fname_coil: Pointing to coil profile. 4-dimensional: x, y, z, coil.
        fname_fmap:
        fname_mask:
        verbose:

    Returns:

    """
    coil = nib.load(fname_coil).get_fdata()
    nii_fmap = nib.load(fname_fmap)
    fieldmap = nii_fmap.get_fdata()

    # TODO: Error handling might move to API
    if fieldmap.ndim != 4:
        raise RuntimeError('fmap must be 4d (x, y, z, t)')
    nx, ny, nz, nt = fieldmap.shape

    # TODO: check good practice below
    if fname_mask is not None:
        mask = nib.load(fname_mask).get_fdata()
    else:
        mask = np.ones_like(fieldmap)

    # Setup coil
    n_coils = coil.shape[-1]
    currents = np.zeros([n_coils, nt])

    shimmed = np.zeros_like(fieldmap)
    masked_fieldmaps = np.zeros_like(fieldmap)
    masked_shimmed = np.zeros_like(shimmed)
    for i_t in range(nt):
        currents[:, i_t] = sequential_zslice(fieldmap[..., i_t], coil, mask, z_slices=np.array(range(nz)),
                                             bounds=[(-np.inf, np.inf)]*8)
        shimmed[..., i_t] = fieldmap[..., i_t] + np.sum(currents[:, i_t] * coil, axis=3, keepdims=False)
        masked_fieldmaps[..., i_t] = mask * fieldmap[..., i_t]
        masked_shimmed[..., i_t] = mask * shimmed[..., i_t]

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
    im = ax.imshow(fieldmap[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Unshimmed")
    ax = fig.add_subplot(2, 2, 4)
    im = ax.imshow(shimmed[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Shimmed")

    click.echo(f"\nThe associated current coefficients are : {currents[:, i_t]}")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_plot.png')
    fig.savefig(fname_figure)

    fig = Figure(figsize=(10, 10))
    # TODO: display time as X -- add label for Y
    for i_t in range(nt):
        ax = fig.add_subplot(nt, 1, i_t + 1)
        ax.plot(np.arange(8), currents[:, i_t])
        ax.set_title(f"Time {i_t}")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_currents.png')
    fig.savefig(fname_figure)

    # TODO: fetch PMU timing
    get_acquisition_time()


    # TODO:
    #  fit PMU and fieldmap values
    #  do regression to separate static componant and RIRO component
    #  output coefficient with proper scaling

    return fname_figure


if __name__ == '__main__':
    fname_coil='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/coil_profile.nii.gz'
    fname_fmap='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/sub-example_fieldmap.nii.gz'
    fname_mask='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/mask.nii.gz'
    realtime_zshim(fname_coil, fname_fmap, fname_mask)
