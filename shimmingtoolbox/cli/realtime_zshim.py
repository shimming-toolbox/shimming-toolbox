#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import numpy as np
import os
import nibabel as nib
import json
from sklearn.linear_model import LinearRegression
from nibabel.processing import resample_from_to
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
@click.option('-anat', 'fname_anat', type=click.Path(),
              help="Filename of the anatomical image to apply the correction.")
# TODO: Remove json file as input
@click.option('-json', 'fname_json', type=click.Path(),
              help="Filename of json corresponding BIDS sidecar.")
@click.option("-verbose", is_flag=True, help="Be more verbose.")
def realtime_zshim(fname_coil, fname_fmap, fname_mask, fname_resp, fname_json, fname_anat, verbose=True):
    """

    Args:
        fname_coil: Pointing to coil profile. 4-dimensional: x, y, z, coil.
        fname_fmap:
        fname_mask:
        fname_resp:
        verbose:

    Returns:

    """
    # Load coil
    # When using only z channel (corresponding to index 0) TODO:Remove
    # coil = np.expand_dims(nib.load(fname_coil).get_fdata()[:, :, :, 0], -1)
    # When using all channels TODO: Keep
    coil = nib.load(fname_coil).get_fdata()

    # Load fieldmap
    nii_fmap = nib.load(fname_fmap)
    fieldmap = nii_fmap.get_fdata()

    # TODO: Error handling might move to API
    if fieldmap.ndim != 4:
        raise RuntimeError("fmap must be 4d (x, y, z, t)")
    nx, ny, nz, nt = fieldmap.shape

    # Load mask
    # TODO: check good practice below
    if fname_mask is not None:
        mask = nib.load(fname_mask).get_fdata()
    else:
        mask = np.ones_like(fieldmap)

    # Load anat
    nii_anat = nib.load(fname_anat)
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise RuntimeError("Anatomical image must be in 3d")

    # Shim using sequencer and optimizer
    n_coils = coil.shape[-1]
    currents = np.zeros([n_coils, nt])
    shimmed = np.zeros_like(fieldmap)
    masked_fieldmaps = np.zeros_like(fieldmap)
    for i_t in range(nt):
        currents[:, i_t] = sequential_zslice(fieldmap[..., i_t], coil, mask, z_slices=np.array(range(nz)),
                                             bounds=[(-np.inf, np.inf)] * n_coils)
        shimmed[..., i_t] = fieldmap[..., i_t] + np.sum(currents[:, i_t] * coil, axis=3, keepdims=False)
        masked_fieldmaps[..., i_t] = mask * fieldmap[..., i_t]

    # Calculate gz gradient
    # Image is z, y, x
    # Pixdim[3] is the space between pixels in the z direction in millimeters
    # TODO: Investigate if axes should be 1 or 0
    g = 1000 / 42.576e6  # [mT / Hz]
    gz_gradient = np.zeros_like(masked_fieldmaps)
    for it in range(nt):
        gz_gradient[..., 0, it] = np.gradient(g * masked_fieldmaps[:, :, 0, it],
                                              nii_fmap.header['pixdim'][3] / 1000,
                                              axis=1)  # [mT / m]

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
    #  field(i_vox) = a(i_vox) * (acq_pressures - mean_p) + b(i_vox)
    #    could use: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    #  Note: strong spatial autocorrelation on the a and b coefficients. Ie: two adjacent voxels are submitted to similar
    #  static B0 field and RIRO component. --> we need to find a way to account for that
    #   solution 1: post-fitting regularization.
    #     pros: easy to implement
    #     cons: fit is less robust to noise
    #   solution 2: accounting for regularization during fitting
    #     pros: fitting more robust to noise
    #     cons: (from Ryan): regularized fitting took a lot of time on Matlab

    # Shim using PMU
    mean_p = np.mean(acq_pressures)
    riro = np.zeros_like(fieldmap[:, :, :, 0])
    static = np.zeros_like(fieldmap[:, :, :, 0])
    for i_x in range(fieldmap.shape[0]):
        for i_y in range(fieldmap.shape[1]):
            for i_z in range(fieldmap.shape[2]):
                # TODO: Fit for -masked_field?
                # reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p, -gz_gradient[i_x, i_y, i_z, :])
                reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p, -masked_fieldmaps[i_x, i_y, i_z, :])
                riro[i_x, i_y, i_z] = reg.coef_
                static[i_x, i_y, i_z] = reg.intercept_

    # Resample masked_fieldmaps, riro and static to target anatomical image
    # TODO: convert to a function
    masked_fmap_4d = np.zeros(anat.shape + (nt,))
    for it in range(nt):
        nii_masked_fmap_3d = nib.Nifti1Image(masked_fieldmaps[..., it], nii_fmap.affine)
        nii_resampled_fmap_3d = resample_from_to(nii_masked_fmap_3d, nii_anat, mode='nearest')
        masked_fmap_4d[..., it] = nii_resampled_fmap_3d.get_fdata()

    nii_resampled_fmap = nib.Nifti1Image(masked_fmap_4d, nii_anat.affine)
    nii_riro = nib.Nifti1Image(riro, nii_fmap.affine)
    nii_static = nib.Nifti1Image(static, nii_fmap.affine)
    nii_resampled_riro = resample_from_to(nii_riro, nii_anat, mode='nearest')
    nii_resampled_static = resample_from_to(nii_static, nii_anat, mode='nearest')

    nib.save(nii_resampled_fmap, os.path.join(__dir_shimmingtoolbox__, 'resampled_fmap.nii.gz'))
    nib.save(nii_resampled_riro, os.path.join(__dir_shimmingtoolbox__, 'resampled_riro.nii.gz'))
    nib.save(nii_resampled_static, os.path.join(__dir_shimmingtoolbox__, 'resampled_static.nii.gz'))

    # Calculate the mean for riro and static for a perticular slice
    n_slices = nii_resampled_fmap.get_fdata().shape[2]
    static_correction = np.zeros([n_slices])
    riro_correction = np.zeros([n_slices])
    for i_slice in range(n_slices):
        static_correction[i_slice] = np.mean(nii_resampled_static.get_fdata()[..., i_slice])
        riro_correction[i_slice] = np.mean(nii_resampled_riro.get_fdata()[..., i_slice])

    # Write to a text file
    # TODO: Add as an option to output the file to a specified location
    fname_corrections = os.path.join(__dir_shimmingtoolbox__, 'zshim_gradients.txt')
    file_gradients = open(fname_corrections, 'w')
    for i_slice in range(n_slices):
        file_gradients.write(f'Vector_Gz[0][{i_slice}]= {static_correction[i_slice]:.6f}\n')
        file_gradients.write(f'Vector_Gz[1][{i_slice}]= {riro_correction[i_slice]:.12f}\n')
        file_gradients.write(f'Vector_Gz[2][{i_slice}]= {mean_p:.3f}\n')
        # Matlab includes the mean pressure
    file_gradients.close()

    # ================ PLOTS ================

    # Calculate masked shim for spherical harmonics plot
    masked_shimmed = np.zeros_like(shimmed)
    for i_t in range(nt):
        masked_shimmed[..., i_t] = mask * shimmed[..., i_t]

    # Plot unshimmed vs shimmed and their mask for spherical harmonics
    i_t = 0
    fig = Figure(figsize=(10, 10))
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
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_sphharm_shimmed.png')
    fig.savefig(fname_figure)

    # Plot the coil coefs through time
    fig = Figure(figsize=(10, 10))
    for i_coil in range(n_coils):
        ax = fig.add_subplot(n_coils, 1, i_coil + 1)
        ax.plot(np.arange(nt), currents[i_coil, :])
        ax.set_title(f"Channel {i_coil}")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_sphharm_currents.png')
    fig.savefig(fname_figure)

    # Plot Static and RIRO
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(riro[:-1, :-1, 0])
    fig.colorbar(im)
    ax.set_title("RIRO")
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(static[:-1, :-1, 0])
    fig.colorbar(im)
    ax.set_title("Static")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_riro_static.png')
    fig.savefig(fname_figure)

    # Calculate fitted and shimmed for pressure fitted plot
    fitted_fieldmap = riro * (acq_pressures -mean_p) + static
    shimmed_pressure_fitted = np.expand_dims(fitted_fieldmap, 2) + masked_fieldmaps

    # Plot pressure fitted fieldmap
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 1, 1)
    im = ax.imshow(masked_fieldmaps[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("fieldmap")
    ax = fig.add_subplot(3, 1, 2)
    im = ax.imshow(fitted_fieldmap[:-1, :-1, i_t])
    fig.colorbar(im)
    ax.set_title("Fit")
    ax = fig.add_subplot(3, 1, 3)
    im = ax.imshow(shimmed_pressure_fitted[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Shimmed (fit + fieldmap")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_pressure_fitted.png')
    fig.savefig(fname_figure)

    # Reshape pmu datapoints to fit those of the acquisition
    pmu_times = np.linspace(pmu.start_time_mdh, pmu.stop_time_mdh, len(pmu.data))
    pmu_times_within_range = pmu_times[pmu_times > acq_timestamps[0]]
    pmu_data_within_range = pmu.data[pmu_times > acq_timestamps[0]]
    pmu_data_within_range = pmu_data_within_range[pmu_times_within_range < acq_timestamps[fieldmap.shape[3] - 1]]
    pmu_times_within_range = pmu_times_within_range[pmu_times_within_range < acq_timestamps[fieldmap.shape[3] - 1]]

    # Calc fieldmap average within mask
    fieldmap_avg = np.zeros([fieldmap.shape[3]])
    for i_time in range(nt):
        masked_array = np.ma.array(fieldmap[:, :, :, i_time], mask=mask == False)
        fieldmap_avg[i_time] = np.ma.average(masked_array)

    # Plot pmu vs B0 in masked region
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    ax.plot(acq_timestamps / 1000, acq_pressures, label='Interpolated pressures')
    # ax.plot(pmu_times / 1000, pmu.data, label='Raw pressures')
    ax.plot(pmu_times_within_range / 1000, pmu_data_within_range, label='Pmu pressures')
    ax.legend()
    ax.set_title("Pressure [0, 4095] vs time (s) ")
    ax = fig.add_subplot(212)
    ax.plot(acq_timestamps / 1000, fieldmap_avg, label='Mean B0')
    ax.legend()
    ax.set_title("Fieldmap average over unmasked region (Hz) vs time (s)")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_pmu_vs_B0.png')
    fig.savefig(fname_figure)

    # Show anatomical image
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(anat[:, :, 10])
    fig.colorbar(im)
    ax.set_title("Anatomical image [:, :, 10]")
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(nii_resampled_fmap.get_fdata()[:, :, 10, 0])
    fig.colorbar(im)
    ax.set_title("Resampled fieldmap [:, :, 10, 0]")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'reatime_zshime_anat.png')
    fig.savefig(fname_figure)

    # Show Gradient
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(gz_gradient[:, :, 0, 0])
    fig.colorbar(im)
    ax.set_title("Gradient [:, :, 0, 0]")
    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'reatime_zshime_gradient.png')
    fig.savefig(fname_figure)

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
