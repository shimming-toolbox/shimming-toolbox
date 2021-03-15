#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import nibabel as nib
from sklearn.linear_model import LinearRegression
from skimage.filters import gaussian
from matplotlib.figure import Figure

from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.utils import st_progress_bar
from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.coils.coordinates import phys_gradient
from shimmingtoolbox.coils.coordinates import phys_to_vox_gradient


def realtime_zshim(nii_fieldmap, nii_anat, pmu, json_fmap, nii_mask_anat_riro=None, nii_mask_anat_static=None,
                   path_output=None):
    """ This function will generate static and dynamic (due to respiration) Gz components based on a fieldmap time
    series and respiratory trace information obtained from Siemens bellows  (PMUresp_signal.resp). An additional
    multi-gradient echo (MGRE) magnitude image is used to generate an ROI and resample the static and dynamic Gz
    component maps to match the MGRE image. Lastly the mean Gz values within the ROI are computed for each slice.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries. Fieldmap should be in Hz.
        nii_anat (nibabel.Nifti1Image):  Nibabel object containing a 3d image of the target data to shim.
        pmu (PmuResp): Filename of the file of the respiratory trace.
        json_fmap (dict): dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        nii_mask_anat_static (nibabel.Nifti1Image): Nibabel object containing the mask to specify the shimming region
                                                    for the static component.
        nii_mask_anat_riro (nibabel.Nifti1Image): Nibabel object containing the mask to specify the shimming region
                                                  for the riro component.
        path_output (str): Path to output figures and temporary variables. If none is provided, no debug output is
                           provided.

    Returns:
        numpy.ndarray: 1D array of the static_correction. The correction is in mT/m for each slice.
        numpy.ndarray: 1D array of the dynamic riro_correction. The correction is in (mT/m)*rms_pressure for each slice.
        float: Average pressure of the pmu
        float: RMS of the pmu pressure
    """

    # Set up output of figures
    is_outputting_figures = False
    if path_output is not None:
        is_outputting_figures = True
        if not os.path.exists(path_output):
            os.makedirs(path_output)

    # Make sure fieldmap has the appropriate dimensions
    fieldmap = nii_fieldmap.get_fdata()
    if fieldmap.ndim != 4:
        raise RuntimeError("fmap must be 4d (x, y, z, t)")
    nx, ny, nz, nt = nii_fieldmap.shape

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise RuntimeError("Anatomical image must be in 3d")

    # Load riro mask
    if nii_mask_anat_riro is not None:
        if not np.all(np.isclose(nii_anat.affine, nii_mask_anat_riro.affine)) or \
                not np.all(nii_mask_anat_riro.shape == nii_anat.shape):
            raise RuntimeError("Mask must have the same shape and affine transformation as anat")
        nii_fmap_3d_temp = nib.Nifti1Image(fieldmap[..., 0], nii_fieldmap.affine)
        nii_mask_fmap_riro = resample_from_to(nii_mask_anat_riro, nii_fmap_3d_temp)
        mask_fmap_riro = nii_mask_fmap_riro.get_fdata()
    else:
        mask_fmap_riro = np.ones_like(fieldmap[..., 0])
        nii_mask_fmap_riro = nib.Nifti1Image(mask_fmap_riro, nii_anat.affine)
        nii_mask_anat_riro = nib.Nifti1Image(np.ones_like(anat), nii_anat.affine)

    # Load static mask
    if nii_mask_anat_static is not None:
        if not np.all(np.isclose(nii_anat.affine, nii_mask_anat_static.affine)) or \
                not np.all(nii_mask_anat_static.shape == nii_anat.shape):
            raise RuntimeError("Mask must have the same shape and affine transformation as anat")
        nii_fmap_3d_temp = nib.Nifti1Image(fieldmap[..., 0], nii_fieldmap.affine)
        nii_mask_fmap_static = resample_from_to(nii_mask_anat_static, nii_fmap_3d_temp)
        mask_fmap_static = nii_mask_fmap_static.get_fdata()
    else:
        mask_fmap_static = np.ones_like(fieldmap[..., 0])
        nii_mask_fmap_static = nib.Nifti1Image(mask_fmap_static, nii_anat.affine)
        nii_mask_anat_static = nib.Nifti1Image(np.ones_like(anat), nii_anat.affine)

    if is_outputting_figures:
        nib.save(nii_mask_fmap_riro, os.path.join(path_output, 'fig_mask_fmap_riro.nii.gz'))
        nib.save(nii_mask_fmap_static, os.path.join(path_output, 'fig_mask_fmap_static.nii.gz'))

    masked_fieldmaps_static = np.zeros_like(fieldmap)
    masked_fieldmaps_riro = np.zeros_like(fieldmap)
    for i_t in range(nt):
        masked_fieldmaps_static[..., i_t] = mask_fmap_static * fieldmap[..., i_t]
        masked_fieldmaps_riro[..., i_t] = mask_fmap_riro * fieldmap[..., i_t]

    # Calculate Gx, Gy and Gz gradients (in the physical coordinate system)
    g = 1000 / 42.5774785178325552e6  # [mT / Hz]
    gradient = np.array([np.zeros_like(fieldmap), np.zeros_like(fieldmap), np.zeros_like(fieldmap)])
    for it in range(nt):
        gradient[:][..., it] = phys_gradient(g * fieldmap[:, :, :, it], nii_fieldmap.affine)  # [mT / mm]
    gradient *= 1000  # [mT / m]

    if is_outputting_figures:
        nii_gz_gradient = nib.Nifti1Image(gradient[2], nii_fieldmap.affine)
        nib.save(nii_gz_gradient, os.path.join(path_output, 'fig_gz_gradient.nii.gz'))
        nii_gy_gradient = nib.Nifti1Image(gradient[1], nii_fieldmap.affine)
        nib.save(nii_gy_gradient, os.path.join(path_output, 'fig_gy_gradient.nii.gz'))
        nii_gx_gradient = nib.Nifti1Image(gradient[0], nii_fieldmap.affine)
        nib.save(nii_gx_gradient, os.path.join(path_output, 'fig_gx_gradient.nii.gz'))

    # Fetch PMU timing
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_fmap)
    # TODO: deal with saturation
    # fit PMU and fieldmap values
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # Shim using PMU
    # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
    # Note: strong spatial autocorrelation on the a and b coefficients. Ie: two adjacent voxels are submitted to similar
    #  static B0 field and RIRO component. --> we need to find a way to account for that
    #   solution 1: post-fitting regularization.
    #     pros: easy to implement
    #     cons: fit is less robust to noise
    #   solution 2: accounting for regularization during fitting
    #     pros: fitting more robust to noise
    #     cons: (from Ryan): regularized fitting took a lot of time on Matlab
    mean_p = np.mean(acq_pressures)
    pressure_rms = np.sqrt(np.mean((acq_pressures - mean_p) ** 2))
    reg = LinearRegression().fit(acq_pressures.reshape(-1, 1) - mean_p,
                                 -gradient.reshape(-1, gradient.shape[-1]).T)
    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    riro = reg.coef_.reshape(gradient.shape[:-1]) * pressure_rms
    static = reg.intercept_.reshape(gradient.shape[:-1])

        #riro[g_axis][:, :, :] = gaussian(riro[g_axis][:, :, :], sigma = 2, mode = 'constant', cval = 0.0,
        # multichannel=False)
        #static[g_axis][:, :, :] = gaussian(static[g_axis][:, :, :], sigma = 1.5, mode = 'constant', cval = 0.0,
        # multichannel=False)

    # Resample static to target anatomical image
    resampled_static = np.array([np.zeros_like(anat), np.zeros_like(anat), np.zeros_like(anat)])
    for g_axis in range(3):
        nii_static = nib.Nifti1Image(static[g_axis], nii_fieldmap.affine)
        nii_resampled_static = resample_from_to(nii_static, nii_anat)
        resampled_static[g_axis] = nii_resampled_static.get_fdata()
    # Since this is zshimming, only the slice component is used.
    _, _, resampled_static_vox = phys_to_vox_gradient(resampled_static[0], resampled_static[1], resampled_static[2],
                                                      nii_anat.affine)
    nii_resampled_static_vox = nib.Nifti1Image(resampled_static_vox, nii_anat.affine)
    nii_resampled_static_masked = nib.Nifti1Image(resampled_static_vox * nii_mask_anat_static.get_fdata(),
                                                  nii_resampled_static_vox.affine)
    if is_outputting_figures:
        nib.save(nii_resampled_static_masked, os.path.join(path_output, 'fig_resampled_static.nii.gz'))

    # Resample riro to target anatomical image
    resampled_riro = np.array([np.zeros_like(anat), np.zeros_like(anat), np.zeros_like(anat)])
    for g_axis in range(3):
        nii_riro = nib.Nifti1Image(riro[g_axis], nii_fieldmap.affine)
        nii_resampled_riro = resample_from_to(nii_riro, nii_anat)
        resampled_riro[g_axis] = nii_resampled_riro.get_fdata()
    # Since this is zshimming, only the slice component is used.
    _, _, resampled_riro_vox = phys_to_vox_gradient(resampled_riro[0], resampled_riro[1], resampled_riro[2],
                                                    nii_anat.affine)
    nii_resampled_riro_vox = nib.Nifti1Image(resampled_riro_vox, nii_anat.affine)
    nii_resampled_static_masked = nib.Nifti1Image(resampled_riro_vox * nii_mask_anat_riro.get_fdata(),
                                                  nii_resampled_riro_vox.affine)
    if is_outputting_figures:
        nib.save(nii_resampled_static_masked, os.path.join(path_output, 'fig_resampled_riro.nii.gz'))

    # Calculate the mean for riro and static for a particular slice
    n_slices = nii_anat.get_fdata().shape[2]
    static_correction = np.zeros([n_slices])
    riro_correction = np.zeros([n_slices])
    for i_slice in range(n_slices):
        ma_static_anat = np.ma.array(resampled_static_vox[..., i_slice],
                                     mask=nii_mask_anat_static.get_fdata()[..., i_slice] == False)
        static_correction[i_slice] = np.ma.mean(ma_static_anat)

        ma_riro_anat = np.ma.array(resampled_riro_vox[..., i_slice],
                                   mask=nii_mask_anat_riro.get_fdata()[..., i_slice] == False)
        riro_correction[i_slice] = np.ma.mean(ma_riro_anat)
    static_correction[np.isnan(static_correction)] = 0.
    riro_correction[np.isnan(riro_correction)] = 0.

    # ================ PLOTS ================

    if is_outputting_figures:

        # Plot Static and RIRO
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        im = ax.imshow(riro[2][:-1, :-1, 0] / pressure_rms)
        fig.colorbar(im)
        ax.set_title("RIRO")
        ax = fig.add_subplot(2, 1, 2)
        im = ax.imshow(static[2][:-1, :-1, 0])
        fig.colorbar(im)
        ax.set_title("Static")
        fname_figure = os.path.join(path_output, 'fig_realtime_zshim_riro_static.png')
        fig.savefig(fname_figure)

        # Reshape pmu datapoints to fit those of the acquisition
        pmu_times = np.linspace(pmu.start_time_mdh, pmu.stop_time_mdh, len(pmu.data))
        pmu_times_within_range = pmu_times[pmu_times > acq_timestamps[0]]
        pmu_data_within_range = pmu.data[pmu_times > acq_timestamps[0]]
        pmu_data_within_range = pmu_data_within_range[pmu_times_within_range < acq_timestamps[fieldmap.shape[3] - 1]]
        pmu_times_within_range = pmu_times_within_range[pmu_times_within_range < acq_timestamps[fieldmap.shape[3] - 1]]

        # Calc fieldmap average within static mask
        fieldmap_avg = np.zeros([fieldmap.shape[3]])
        for i_time in range(nt):
            masked_array = np.ma.array(fieldmap[:, :, :, i_time], mask=mask_fmap_static == False)
            fieldmap_avg[i_time] = np.ma.average(masked_array)

        # Plot pmu vs B0 in static masked region
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
        ax.set_title("Fieldmap average over static unmasked region (Hz) vs time (s)")
        fname_figure = os.path.join(path_output, 'fig_realtime_zshim_pmu_vs_B0.png')
        fig.savefig(fname_figure)

        # Show anatomical image and masks
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 3, 1)
        im = ax.imshow(anat[:, :, 10])
        fig.colorbar(im)
        ax.set_title("Anatomical image [:, :, 10]")
        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(nii_mask_anat_static.get_fdata()[:, :, 10])
        fig.colorbar(im)
        ax.set_title("Mask static [:, :, 10]")
        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(nii_mask_anat_riro.get_fdata()[:, :, 10])
        fig.colorbar(im)
        ax.set_title("Mask riro [:, :, 10]")
        fname_figure = os.path.join(path_output, 'fig_reatime_zshim_anat_mask.png')
        fig.savefig(fname_figure)

        # Show Gradient
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(gradient[2][:, :, 0, 0])
        fig.colorbar(im)
        ax.set_title("Z Gradient [:, :, 0, 0]")
        fname_figure = os.path.join(path_output, 'fig_realtime_zshim_zgradient.png')
        fig.savefig(fname_figure)

        # Show evolution of coefficients
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(range(n_slices), static_correction, label='Static correction')
        ax.set_title("Static correction evolution through slices")
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(range(n_slices), (acq_pressures.max() - mean_p) * (riro_correction / pressure_rms),
                label='Riro correction')
        ax.set_title("Riro correction evolution through slices")
        fname_figure = os.path.join(path_output, 'fig_realtime_zshim_correction_slice.png')
        fig.savefig(fname_figure)

    return static_correction, riro_correction, mean_p, pressure_rms
