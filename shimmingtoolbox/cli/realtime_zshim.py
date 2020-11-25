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
@click.option('-mask', 'fname_mask_anat', type=click.Path(),
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
def realtime_zshim(fname_fmap, fname_mask_anat, fname_resp, fname_json, fname_anat, fname_output, verbose=True):
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

    # Look if output directory exists, if not, create it
    if not os.path.exists(fname_output):
        os.makedirs(fname_output)

    # Load fieldmap
    nii_fmap = nib.load(fname_fmap)
    fieldmap = nii_fmap.get_fdata()

    # TODO: Error handling might move to API
    if fieldmap.ndim != 4:
        raise RuntimeError("fmap must be 4d (x, y, z, t)")
    nx, ny, nz, nt = fieldmap.shape

    # Load anat
    nii_anat = nib.load(fname_anat)
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise RuntimeError("Anatomical image must be in 3d")

    # Load mask
    # TODO: check good practice below
    if fname_mask_anat is not None:
        nii_mask_anat = nib.load(fname_mask_anat)
        if not np.all(np.isclose(nii_anat.affine, nii_mask_anat.affine)) or\
                not np.all(nii_mask_anat.shape == nii_anat.shape):
            raise RuntimeError("Mask must have the same shape and affine transformation as anat")
        nii_fmap_3d_temp = nib.Nifti1Image(fieldmap[..., 0], nii_fmap.affine)
        nii_mask_fmap = resample_from_to(nii_mask_anat, nii_fmap_3d_temp)
        mask_fmap = nii_mask_fmap.get_fdata()
    else:
        mask_fmap = np.ones_like(fieldmap)
        nii_mask_fmap = nib.Nifti1Image(mask_fmap, nii_anat.affine)
        nii_mask_anat = nib.Nifti1Image(np.ones_like(anat), nii_anat.affine)

    if DEBUG:
        nib.save(nii_mask_fmap, os.path.join(fname_output, 'tmp.mask_fmap_resample.nii.gz'))

    masked_fieldmaps = np.zeros_like(fieldmap)
    for i_t in range(nt):
        masked_fieldmaps[..., i_t] = mask_fmap * fieldmap[..., i_t]

    # Calculate gz gradient
    g = 1000 / 42.576e6  # [mT / Hz]
    gz_gradient = np.zeros_like(fieldmap)
    # Get voxel coordinates. Z coordinates correspond to coord[2]
    z_coord = generate_meshgrid(mask_fmap.shape, nii_fmap.affine)[2] / 1000  # [m]

    for it in range(nt):
        gz_gradient[..., 0, it] = np.gradient(g * fieldmap[:, :, 0, it], z_coord[0, :, 0], axis=1)  # [mT / m]
    if DEBUG:
        nii_gz_gradient = nib.Nifti1Image(gz_gradient, nii_fmap.affine)
        nib.save(nii_gz_gradient, os.path.join(fname_output, 'tmp.gz_gradient.nii.gz'))

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
    pressure_rms = np.sqrt(np.mean((acq_pressures - mean_p) ** 2))
    riro = np.zeros_like(fieldmap[:, :, :, 0])
    static = np.zeros_like(fieldmap[:, :, :, 0])
    # TODO fix progress bar not showing up
    progress_bar = st_progress_bar(fieldmap[..., 0].size, desc="Fitting", ascii=False)
    for i_x in range(fieldmap.shape[0]):
        for i_y in range(fieldmap.shape[1]):
            for i_z in range(fieldmap.shape[2]):
                reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p, -gz_gradient[i_x, i_y, i_z, :])
                # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
                # between scans. This allows to compare results between scans.
                riro[i_x, i_y, i_z] = reg.coef_ * pressure_rms
                static[i_x, i_y, i_z] = reg.intercept_
                progress_bar.update(1)

    # Resample masked_fieldmaps to target anatomical image
    # TODO: convert to a function
    masked_fmap_4d = np.zeros(anat.shape + (nt,))
    for it in range(nt):
        nii_masked_fmap_3d = nib.Nifti1Image(masked_fieldmaps[..., it], nii_fmap.affine)
        nii_resampled_fmap_3d = resample_from_to(nii_masked_fmap_3d, nii_anat, order=2, mode='nearest')
        masked_fmap_4d[..., it] = nii_resampled_fmap_3d.get_fdata()
    nii_resampled_fmap = nib.Nifti1Image(masked_fmap_4d, nii_anat.affine)

    if DEBUG:
        nib.save(nii_resampled_fmap, os.path.join(fname_output, 'resampled_fmap.nii.gz'))

    # Resample static to target anatomical image
    nii_static = nib.Nifti1Image(static, nii_fmap.affine)
    nii_resampled_static = resample_from_to(nii_static, nii_anat, mode='nearest')
    nii_resampled_static_masked = nib.Nifti1Image(nii_resampled_static.get_fdata() * nii_mask_anat.get_fdata(),
                                                  nii_resampled_static.affine)
    if DEBUG:
        nib.save(nii_resampled_static_masked, os.path.join(fname_output, 'resampled_static.nii.gz'))

    # Resample riro to target anatomical image
    nii_riro = nib.Nifti1Image(riro, nii_fmap.affine)
    nii_resampled_riro = resample_from_to(nii_riro, nii_anat, mode='nearest')
    nii_resampled_riro_masked = nib.Nifti1Image(nii_resampled_riro.get_fdata() * nii_mask_anat.get_fdata(),
                                                nii_resampled_riro.affine)
    if DEBUG:
        nib.save(nii_resampled_riro_masked, os.path.join(fname_output, 'resampled_riro.nii.gz'))

    # Calculate the mean for riro and static for a perticular slice
    n_slices = nii_anat.get_fdata().shape[2]
    static_correction = np.zeros([n_slices])
    riro_correction = np.zeros([n_slices])
    for i_slice in range(n_slices):
        ma_static_anat = np.ma.array(nii_resampled_static.get_fdata()[..., i_slice],
                                     mask=nii_mask_anat.get_fdata()[..., i_slice] == False)
        static_correction[i_slice] = np.ma.mean(ma_static_anat)
        ma_riro_anat = np.ma.array(nii_resampled_riro.get_fdata()[..., i_slice],
                                   mask=nii_mask_anat.get_fdata()[..., i_slice] == False)
        riro_correction[i_slice] = np.ma.mean(ma_riro_anat) / pressure_rms

    # Write to a text file
    fname_corrections = os.path.join(fname_output, 'zshim_gradients.txt')
    file_gradients = open(fname_corrections, 'w')
    for i_slice in range(n_slices):
        file_gradients.write(f'Vector_Gz[0][{i_slice}]= {static_correction[i_slice]:.6f}\n')
        file_gradients.write(f'Vector_Gz[1][{i_slice}]= {riro_correction[i_slice]:.12f}\n')
        file_gradients.write(f'Vector_Gz[2][{i_slice}]= {mean_p:.3f}\n')
    file_gradients.close()

    # ================ PLOTS ================

    if DEBUG:

        # Plot Static and RIRO
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        im = ax.imshow(riro[:-1, :-1, 0] / pressure_rms)
        fig.colorbar(im)
        ax.set_title("RIRO")
        ax = fig.add_subplot(2, 1, 2)
        im = ax.imshow(static[:-1, :-1, 0])
        fig.colorbar(im)
        ax.set_title("Static")
        fname_figure = os.path.join(fname_output, 'fig_realtime_zshim_riro_static.png')
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
            masked_array = np.ma.array(fieldmap[:, :, :, i_time], mask=mask_fmap == False)
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
        fname_figure = os.path.join(fname_output, 'fig_realtime_zshim_pmu_vs_B0.png')
        fig.savefig(fname_figure)

        # Show anatomical image
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        im = ax.imshow(anat[:, :, 10])
        fig.colorbar(im)
        ax.set_title("Anatomical image [:, :, 10]")
        ax = fig.add_subplot(2, 1, 2)
        im = ax.imshow(nii_mask_anat.get_fdata()[:, :, 10])
        fig.colorbar(im)
        ax.set_title("Mask [:, :, 10]")
        fname_figure = os.path.join(fname_output, 'fig_reatime_zshim_anat.png')
        fig.savefig(fname_figure)

        # Show Gradient
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(gz_gradient[:, :, 0, 0])
        fig.colorbar(im)
        ax.set_title("Gradient [:, :, 0, 0]")
        fname_figure = os.path.join(fname_output, 'fig_realtime_zshim_gradient.png')
        fig.savefig(fname_figure)

        # Show evolution of coefficients
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(range(n_slices), static_correction, label='Static correction')
        ax.set_title("Static correction evolution through slices")
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(range(n_slices), (acq_pressures.max() - mean_p) * riro_correction, label='Riro correction')
        ax.set_title("Riro correction evolution through slices")
        fname_figure = os.path.join(fname_output, 'fig_realtime_zshim_correction_slice.png')
        fig.savefig(fname_figure)

    return fname_corrections
