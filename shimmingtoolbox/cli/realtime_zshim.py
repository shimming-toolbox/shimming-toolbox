#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
import json
# TODO: remove matplotlib and dirtesting import
from matplotlib.figure import Figure
from shimmingtoolbox import __dir_testing__

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox import __dir_shimmingtoolbox__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help=f"Perform realtime z-shimming."
)
@click.option('-coil', 'fname_coil', required=True, type=click.Path(),
              help="Coil basis to use for shimming. Enter multiple files if "
                   "you wish to use more than one set of shim coils (eg: "
                   "Siemens gradient/shim coils and external custom coils).")
@click.option('-fmap', 'fname_fmap', required=True, type=click.Path(),
              help="B0 fieldmap. For realtime shimming, this should be a 4d file (4th dimension being time")
@click.option('-mask', 'fname_mask', type=click.Path(),
              help="3D nifti file with voxels between 0 and 1 used to weight the spatial region to shim.")
@click.option('-resp', 'fname_resp', type=click.Path(),
              help="Siemens respiratory file containing pressure data.")
# TODO: Remove json file as input
@click.option('-json', 'fname_json', type=click.Path(),
              help="Filename of json corresponding BIDS sidecar.")
@click.option("-verbose", is_flag=True, help="Be more verbose.")
def realtime_zshim(fname_coil, fname_fmap, fname_mask, fname_resp, fname_json, verbose=True):
    """

    Args:
        fname_coil: Pointing to coil profile. 4-dimensional: x, y, z, coil.
        fname_fmap:
        fname_mask:
        fname_resp:
        verbose:

    Returns:

    """
    # When using only z channnel (corresponding to the 2nd index) TODO:Remove
    # coil = np.expand_dims(nib.load(fname_coil).get_fdata()[:, :, :, 2], -1)

    # When using all channels TODO: Keep
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
                                             bounds=[(-np.inf, np.inf)]*n_coils)
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
    for i_coil in range(n_coils):
        ax = fig.add_subplot(n_coils, 1, i_coil + 1)
        ax.plot(np.arange(nt), currents[i_coil, :])
        ax.set_title(f"Channel {i_coil}")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_currents.png')
    fig.savefig(fname_figure)

    # Fetch PMU timing
    # TODO: Add json to fieldmap instead of asking for another json file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)
    acq_timestamps = get_acquisition_times(nii_fmap, json_data)
    pmu = PmuResp(fname_resp)
    # TODO: deal with saturation
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # TODO:
    #  fit PMU and fieldmap values
    #  do regression to separate static componant and RIRO component
    #  output coefficient with proper scaling
    #  field(i_vox) = a(i_vox) * acq_pressures + b(i_vox)
    #    could use: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    #  Note: strong spatial autocorrelation on the a and b coefficients. Ie: two adjacent voxels are submitted to similar
    #  static B0 field and RIRO component. --> we need to find a way to account for that
    #   solution 1: post-fitting regularization.
    #     pros: easy to implement
    #     cons: fit is less robust to noise
    #   solution 2: accounting for regularization during fitting
    #     pros: fitting more robust to noise
    #     cons: (from Ryan): regularized fitting took a lot of time on Matlab

    #
    return fname_figure

# Debug
# fname_coil = os.path.join(__dir_testing__, 'test_realtime_zshim', 'coil_profile.nii.gz')
# fname_fmap = os.path.join(__dir_testing__, 'test_realtime_zshim', 'sub-example_fieldmap.nii.gz')
# fname_mask = os.path.join(__dir_testing__, 'test_realtime_zshim', 'mask.nii.gz')
# fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
# fname_json = os.path.join(__dir_testing__, 'test_realtime_zshim', 'sub-example_magnitude1.json')
# # fname_coil='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/coil_profile.nii.gz'
# # fname_fmap='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/sub-example_fieldmap.nii.gz'
# # fname_mask='/Users/julien/code/shimming-toolbox/shimming-toolbox/test_realtime_zshim/mask.nii.gz'
# realtime_zshim(fname_coil, fname_fmap, fname_mask, fname_resp, fname_json)
