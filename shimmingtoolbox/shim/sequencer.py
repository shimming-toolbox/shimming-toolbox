#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression
import nibabel as nib
import logging
from nibabel.affines import apply_affine
import os
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.masking.mask_utils import resample_mask

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

supported_optimizers = {
    'least_squares': LsqOptimizer,
    'pseudo_inverse': Optimizer
}


def shim_sequencer(nii_fieldmap, nii_anat, nii_mask_anat, slices, coils: ListCoil, method='least_squares',
                   mask_dilation_kernel='sphere', mask_dilation_kernel_size=3, path_output=os.curdir):
    """
    Performs shimming according to slices using one of the supported optimizers and coil profiles.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 3d and an affine transformation.
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_mask_anat (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim in the region of interest.
                                             (only consider voxels with non-zero values)
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape of anat
                       is: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil profiles
                          are larger than the extent of the fieldmap. This is especially true for dimensions with only
                          1 voxel(e.g. (50x50x1). Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                          :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'. Note: refer to their specific
                      implementation to know limits of the methods in: :mod:`shimmingtoolbox.optimizer`
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output them.

    Returns:
        numpy.ndarray: Coefficients of the coil profiles to shim (len(slices) x n_channels)
    """

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fieldmap.get_fdata().ndim != 3:
        raise ValueError("Fieldmap must be 3d (dim1, dim2, dim3)")
    fieldmap_shape = nii_fieldmap.get_fdata().shape
    # Extend the fieldmap if there are axes that are 1d
    if 1 in fieldmap_shape:
        list_axis = [i for i in range(len(fieldmap_shape)) if fieldmap_shape[i] == 1]
        n_slices = int(math.ceil((mask_dilation_kernel_size - 1) / 2))
        for i_axis in list_axis:
            nii_fieldmap = extend_slice(nii_fieldmap, n_slices=n_slices, axis=i_axis)
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise ValueError("Anatomical image must be in 3d")

    # Make sure the mask has the appropriate dimensions
    mask = nii_mask_anat.get_fdata()
    if mask.ndim != 3:
        raise ValueError("Mask image must be in 3d")

    # Make sure shape and affine of mask are the same as the anat
    if not np.all(mask.shape == anat.shape):
        raise ValueError(f"Shape of mask:\n {mask.shape} must be the same as the shape of anat:\n{anat.shape}")
    if not np.all(nii_mask_anat.affine == nii_anat.affine):
        raise ValueError(f"Affine of mask:\n{nii_mask_anat.affine}\nmust be the same as the affine of anat:\n"
                         f"{nii_anat.affine}")

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, fieldmap, affine_fieldmap, coils)

    # Optimize slice by slice
    coef = _optimize(optimizer, nii_mask_anat, slices, dilation_kernel=mask_dilation_kernel,
                     dilation_size=mask_dilation_kernel_size, path_output=path_output)

    # Evaluate theoretical shim
    _eval_static_shim(optimizer, nii_fieldmap, nii_mask_anat, coef, slices, path_output)

    return coef


def _eval_static_shim(opt: Optimizer, nii_fieldmap, nii_mask, coef, slices, path_output):
    # Calculate theoretical shimmed map
    unshimmed = nii_fieldmap.get_fdata()
    correction_per_channel = np.zeros(opt.merged_coils.shape + (len(slices),))
    shimmed = np.zeros(unshimmed.shape + (len(slices),))
    mask_fieldmap = np.zeros(unshimmed.shape + (len(slices),))
    for i_shim in range(len(slices)):
        correction_per_channel[..., i_shim] = coef[i_shim] * opt.merged_coils
        correction = np.sum(correction_per_channel[..., i_shim], axis=3, keepdims=False)
        shimmed[..., i_shim] = unshimmed + correction

        mask_fieldmap[..., i_shim] = resample_mask(nii_mask, nii_fieldmap, slices[i_shim]).get_fdata()

        sum_shimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * shimmed[..., i_shim]))
        sum_unshimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * unshimmed))

        if sum_shimmed > sum_unshimmed:
            logger.warning("Verify the shim parameters. Some give worse results than no shim.\n"
                           f"i_shim: {i_shim}")

        logger.debug(f"Slice(s): {slices[i_shim]}\n"
                     f"unshimmed: {sum_unshimmed}, shimmed: {sum_shimmed}, current: \n{coef[i_shim, :]}")

    if logger.level <= getattr(logging, 'DEBUG'):
        _plot_currents(coef, path_output)

        # Save correction
        fname_correction = os.path.join(path_output, 'fig_correction_per_channel.nii.gz')
        nii_correction = nib.Nifti1Image(correction_per_channel, opt.unshimmed_affine)
        nib.save(nii_correction, fname_correction)

        fname_correction = os.path.join(path_output, 'fig_shimmed_4thdim_ishim.nii.gz')
        nii_correction = nib.Nifti1Image(correction_per_channel, opt.unshimmed_affine)
        nib.save(nii_correction, fname_correction)

        # Save coils
        nii_merged_coils = nib.Nifti1Image(opt.merged_coils, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_merged_coils, os.path.join(path_output, "merged_coils.nii.gz"))


def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices,
                                pmu: PmuResp, coils: ListCoil, opt_method='least_squares',
                                mask_dilation_kernel='sphere', mask_dilation_kernel_size=3, path_output=os.curdir):
    """
    Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

    Args:
        nii_fieldmap (nibabel.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension is the
                                            timeseries. Also contains an affine transformation.
        json_fmap (dict): Dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                          timestamps).
        nii_anat (nibabel.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_static_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                               component.
        nii_riro_mask (nibabel.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                             component.
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat where the shape of anat:
                       (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        pmu (PmuResp): PmuResp object containing the respiratory trace information.
        coils (ListCoil): List of `Coils` containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil profiles
                          are larger than the extent of the fieldmap. This is especially true for dimensions with only
                          1 voxel(e.g. (50x50x1x10). Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                          :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
        opt_method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'. Note: refer to their specific
                          implementation to know limits of the methods in: :mod:`shimmingtoolbox.optimizer`
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output them.

    Returns:
        (tuple): tuple containing:

            * numpy.ndarray: Static coefficients of the coil profiles to shim (len(slices) x channels) e.g. [Hz]
            * numpy.ndarray: Riro coefficients of the coil profiles to shim (len(slices) x channels)
                             e.g. [Hz/unit_pressure]
            * float: Mean pressure of the respiratory trace.
            * float: Root mean squared of the pressure. This is provided to compare results between scans, multiply the
                     riro coefficients by rms of the pressure to do so.
    """
    # Note: We technically dont need the anat if we use the nii_mask. However, this is a nice safety check to make sure
    # the mask is indeed in the dimension of the anat and not the fieldmap.

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fieldmap.get_fdata().ndim != 4:
        raise ValueError("Fieldmap must be 4d (dim1, dim2, dim3, t)")
    fieldmap_shape = nii_fieldmap.get_fdata().shape[:3]
    # Extend the fieldmap if there are axes that are 1d
    if 1 in fieldmap_shape:
        list_axis = [i for i in range(len(fieldmap_shape)) if fieldmap_shape[i] == 1]
        for i_axis in list_axis:
            n_slices = int(math.ceil((mask_dilation_kernel_size - 1) / 2))
            nii_fieldmap = extend_slice(nii_fieldmap, n_slices=n_slices, axis=i_axis)
    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise ValueError("Anatomical image must be in 3d")

    # Make sure masks have the appropriate dimensions
    static_mask = nii_static_mask.get_fdata()
    if static_mask.ndim != 3:
        raise ValueError("static_mask image must be in 3d")
    riro_mask = nii_riro_mask.get_fdata()
    if riro_mask.ndim != 3:
        raise ValueError("riro_mask image must be in 3d")

    # Make sure shape and affine of masks are the same as the anat
    if not (np.all(riro_mask.shape == anat.shape) and np.all(static_mask.shape == anat.shape)):
        raise ValueError(f"Shape of riro mask: {riro_mask.shape} and static mask: {static_mask.shape} "
                         f"must be the same as the shape of anat: {anat.shape}")
    if not(np.all(nii_riro_mask.affine == nii_anat.affine) and np.all(nii_static_mask.affine == nii_anat.affine)):
        raise ValueError(f"Affine of riro mask:\n{nii_riro_mask.affine}\nand static mask: {nii_static_mask.affine}\n"
                         f"must be the same as the affine of anat:\n{nii_anat.affine}")

    # Fetch PMU timing
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_fmap)
    # TODO: deal with saturation
    # fit PMU and fieldmap values
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # regularization --> static, riro
    # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
    mean_p = np.mean(acq_pressures)
    pressure_rms = np.sqrt(np.mean((acq_pressures - mean_p) ** 2))
    x = acq_pressures.reshape(-1, 1) - mean_p

    # Safety check for linear regression if the pressure and fieldmap fits
    # TODO: Validate the code is indeed a good assessment
    # It should work since I am expecting taking the mask and dilating all the slices should include all the slices
    # resampled and dilated individually
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # Mask the voxels not being shimmed for static
    nii_3dfmap = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine, header=nii_fieldmap.header)
    fmap_mask_static = resample_mask(nii_static_mask, nii_3dfmap, tuple(range(anat.shape[2])),
                                     dilation_kernel=mask_dilation_kernel,
                                     dilation_size=mask_dilation_kernel_size).get_fdata()
    masked_fieldmap_static = np.repeat(fmap_mask_static[..., np.newaxis], fieldmap.shape[-1], 3) * fieldmap
    y = masked_fieldmap_static.reshape(-1, fieldmap.shape[-1]).T

    # Warn if lower than a threshold?
    reg_static = LinearRegression().fit(x, y)
    logger.debug(f"Linear fit of the static masked fieldmap and pressure got a R2 score of: "
                 f"{reg_static.score(x, y)}")

    # Mask the voxels not being shimmed for riro
    fmap_mask_riro = resample_mask(nii_riro_mask, nii_3dfmap, tuple(range(anat.shape[2])),
                                   dilation_kernel=mask_dilation_kernel,
                                   dilation_size=mask_dilation_kernel_size).get_fdata()
    masked_fieldmap_riro = np.repeat(fmap_mask_riro[..., np.newaxis], fieldmap.shape[-1], 3) * fieldmap
    y = masked_fieldmap_riro.reshape(-1, fieldmap.shape[-1]).T
    # Warn if lower than a threshold?
    reg_riro = LinearRegression().fit(x, y)
    logger.debug(f"Linear fit of the riro masked fieldmap and pressure got a R2 score of: "
                 f"{reg_riro.score(x, y)}")

    # Fit to the linear model
    y = fieldmap.reshape(-1, fieldmap.shape[-1]).T
    reg = LinearRegression().fit(x, y)

    # TODO: Safety check for linear regression if the pressure and fieldmap fits, also make sure the above code works
    # It should work since I am expecting taking the mask and dilating all the slices should include all the slices
    # resampled and dilated individually
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # Warn if lower than a threshold?
    logger.debug(f"Linear fit of the fieldmap and pressure got a R2 score of: {reg.score(x, y)}")

    # static/riro contains a 3d matrix of static/riro map in the fieldmap space considering the previous equation
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine_fieldmap, coils)
    coef_static = _optimize(optimizer, nii_static_mask, slices,
                            dilation_kernel=mask_dilation_kernel,
                            dilation_size=mask_dilation_kernel_size,
                            path_output=path_output)

    # Use the currents to define a list of new coil bounds for the riro optimization
    bounds = new_bounds_from_currents(coef_static, optimizer.merged_bounds)

    # Riro shim
    # We multiply by the max offset of the siemens pmu e.g. [max - min = 4095] so that the bounds take effect on the
    # maximum value that the pressure probe can acquire. The equation "riro(i_vox) * (acq_pressures - mean_p)" becomes
    # "riro(i_vox) * max_offset" which is the maximum riro shim we will have. We solve for that to make sure the coils
    # can support it. The units of riro * max_offset are: [unit_shim], ex: [Hz]
    max_offset = max((pmu.max - pmu.min) - mean_p, mean_p)

    # Set the riro map to shim
    # TODO: make sure max_offset could not bust with negative offset
    optimizer.set_unshimmed(riro * max_offset, affine_fieldmap)
    coef_max_riro = _optimize(optimizer, nii_riro_mask, slices,
                              shimwise_bounds=bounds,
                              dilation_kernel=mask_dilation_kernel,
                              dilation_size=mask_dilation_kernel_size,
                              path_output=path_output)
    # Once the coefficients are solved, we divide by max_offset to return to units of
    # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]
    coef_riro = coef_max_riro / max_offset

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    # coef_riro_rms = coef_riro * pressure_rms
    # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

    # Evaluate theoretical shim
    _eval_rt_shim(optimizer, nii_fieldmap, nii_static_mask, coef_static, coef_riro, mean_p,
                  acq_pressures, slices, pressure_rms, pmu, path_output)

    return coef_static, coef_riro, mean_p, pressure_rms


def _eval_rt_shim(opt: Optimizer, nii_fieldmap, nii_mask_static, coef_static, coef_riro, mean_p,
                  acq_pressures, slices, pressure_rms, pmu: PmuResp, path_output):

    logger.debug("Calculating the sum of the shimmed vs unshimmed in the static ROI.")

    # Calculate theoretical shimmed map
    # shim
    unshimmed = nii_fieldmap.get_fdata()
    nii_target = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine, header=nii_fieldmap.header)
    shape = unshimmed.shape + (len(slices),)
    shimmed_static_riro = np.zeros(shape)
    shimmed_static = np.zeros(shape)
    shimmed_riro = np.zeros(shape)
    masked_shim_static_riro = np.zeros(shape)
    masked_shim_static = np.zeros(shape)
    masked_shim_riro = np.zeros(shape)
    masked_unshimmed = np.zeros(shape)
    masked_fieldmap = np.zeros(unshimmed[..., 0].shape + (len(slices),))
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(coef_static[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        # Calculate the riro coil profiles
        riro_profile = np.sum(coef_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        masked_fieldmap[..., i_shim] = resample_mask(nii_mask_static, nii_target, slices[i_shim]).get_fdata()
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
            shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
            shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

            # Calculate masked shim
            masked_shim_static[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static[..., i_t, i_shim]
            masked_shim_static_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static_riro[..., i_t, i_shim]
            masked_shim_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_riro[..., i_t, i_shim]
            masked_unshimmed[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * unshimmed[..., i_t]

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(masked_shim_static[..., i_t, i_shim]))
            sum_shimmed_static_riro = np.sum(np.abs(masked_shim_static_riro[..., i_t, i_shim]))
            sum_shimmed_riro = np.sum(np.abs(masked_shim_riro[..., i_t, i_shim]))
            sum_unshimmed = np.sum(np.abs(masked_unshimmed[..., i_t, i_shim]))

            if sum_shimmed_static_riro > sum_unshimmed:
                # TODO: Remove if too many
                logger.warning("Verify the shim parameters. Some give worse results than no shim.\n"
                               f"i_shim: {i_shim}, i_t: {i_t}")

            logger.debug(f"\ni_shim: {i_shim}, t: {i_t}"
                         f"\nunshimmed: {sum_unshimmed}, shimmed static: {sum_shimmed_static}, "
                         f"shimmed static+riro: {sum_shimmed_static_riro}\n"
                         f"Static currents:\n{coef_static[i_shim]}\n"
                         f"Riro currents:\n{coef_riro[i_shim] * (acq_pressures[i_t] - mean_p)}\n")

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

    # reshape to slice x timepoint
    nt = unshimmed.shape[3]
    n_shim = len(slices)
    shim_trace_static = np.array(shim_trace_static).reshape(n_shim, nt)
    shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_shim, nt)
    shim_trace_riro = np.array(shim_trace_riro).reshape(n_shim, nt)
    unshimmed_trace = np.array(unshimmed_trace).reshape(n_shim, nt)

    # plot results
    i_slice = 0
    i_shim = 0
    i_t = 0
    while np.all(masked_unshimmed[..., i_slice, i_t, i_shim] == np.zeros(masked_unshimmed.shape[:2])):
        i_shim += 1
        if i_shim >= n_shim - 1:
            break

    if logger.level <= getattr(logging, 'DEBUG'):
        _plot_static_riro(masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed, shimmed_static,
                          shimmed_static_riro, path_output, i_slice=i_slice, i_shim=i_shim, i_t=i_t)
        _plot_currents(coef_static, path_output, riro=coef_riro * pressure_rms)
        _plot_shimmed_trace(unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro,
                            path_output)
        _plot_pressure_points(acq_pressures, (pmu.min, pmu.max), path_output)
        _print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, masked_fieldmap)

        # Save shimmed result
        nii_shimmed_static_riro = nib.Nifti1Image(shimmed_static_riro, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_shimmed_static_riro, os.path.join(path_output,
                                                       'shimmed_static_riro_4thdim_it_5thdim_ishim.nii.gz'))

        # Save coils
        nii_merged_coils = nib.Nifti1Image(opt.merged_coils, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_merged_coils, os.path.join(path_output, "merged_coils.nii.gz"))


def _plot_static_riro(masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed, shimmed_static,
                      shimmed_static_riro, path_output, i_t=0, i_slice=0, i_shim=0):
    """Plot Static and RIRO fieldmap for a perticular fieldmap slice, anat shim and timepoint"""

    min_value = min(masked_shim_static_riro[..., i_slice, i_t, i_shim].min(),
                    masked_shim_static[..., i_slice, i_t, i_shim].min(),
                    masked_unshimmed[..., i_slice, i_t, i_shim].min())
    max_value = max(masked_shim_static_riro[..., i_slice, i_t, i_shim].max(),
                    masked_shim_static[..., i_slice, i_t, i_shim].max(),
                    masked_unshimmed[..., i_slice, i_t, i_shim].max())

    fig = Figure(figsize=(15, 10))
    fig.suptitle(f"Maps for shim: {i_shim}, slice: {i_slice}, timepoint: {i_t}")
    ax = fig.add_subplot(2, 3, 1)
    im = ax.imshow(np.rot90(masked_shim_static_riro[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
    fig.colorbar(im)
    ax.set_title("masked_shim static + riro")
    ax = fig.add_subplot(2, 3, 2)
    im = ax.imshow(np.rot90(masked_shim_static[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
    fig.colorbar(im)
    ax.set_title("masked_shim static")
    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(np.rot90(masked_unshimmed[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
    fig.colorbar(im)
    ax.set_title("masked_unshimmed")

    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(np.rot90(shimmed_static_riro[..., i_slice, i_t, i_shim]))
    fig.colorbar(im)
    ax.set_title("shim static + riro")
    ax = fig.add_subplot(2, 3, 5)
    im = ax.imshow(np.rot90(shimmed_static[..., i_slice, i_t, i_shim]))
    fig.colorbar(im)
    ax.set_title(f"shim static")
    ax = fig.add_subplot(2, 3, 6)
    im = ax.imshow(np.rot90(unshimmed[..., i_slice, i_t]))
    fig.colorbar(im)
    ax.set_title(f"unshimmed")
    fname_figure = os.path.join(path_output, 'fig_realtime_masked_shimmed_vs_unshimmed.png')
    fig.savefig(fname_figure)
    logger.debug(f"Saved figure: {fname_figure}")


def _plot_currents(static, path_output: str, riro=None):
    """Plot evolution of currents through shims"""
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_channels = static.shape[1]
    for i_channel in range(n_channels):
        ax.plot(static[:, i_channel], label=f"Static channel{i_channel} currents through shims")
    if riro is not None:
        for i_channel in range(n_channels):
            ax.plot(riro[:, i_channel], label=f"Riro channel{i_channel} currents through shims")
    ax.set_xlabel('i_shims')
    ax.set_ylabel('Coefficients')
    ax.legend()
    ax.set_title("Currents through shims")
    fname_figure = os.path.join(path_output, 'fig_currents.png')
    fig.savefig(fname_figure)
    logger.debug(f"Saved figure: {fname_figure}")


def _plot_pressure_points(acq_pressures, ylim, path_output):
    """Plot respiratory trace pressure points"""
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(acq_pressures, label='pressures')
    ax.legend()
    ax.set_ylim(ylim)
    ax.set_title("Pressures vs time points")
    fname_figure = os.path.join(path_output, 'fig_trace_pressures.png')
    fig.savefig(fname_figure)
    logger.debug(f"Saved figure: {fname_figure}")


def _plot_shimmed_trace(unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro, path_output):
    """plot shimmed and unshimmed sum over the roi for each shim"""

    min_value = min(
        shim_trace_static_riro[:, :].min(),
        shim_trace_static[:, :].min(),
        shim_trace_riro[:, :].min(),
        unshimmed_trace[:, :].min()
    )
    max_value = max(
        shim_trace_static_riro[:, :].max(),
        shim_trace_static[:, :].max(),
        shim_trace_riro[:, :].max(),
        unshimmed_trace[:, :].max()
    )

    # Calc ysize
    n_shims = len(unshimmed_trace)
    ysize = n_shims * 4.7
    fig = Figure(figsize=(10, ysize), tight_layout=True)
    for i_shim in range(n_shims):
        ax = fig.add_subplot(n_shims, 1, i_shim + 1)
        ax.plot(shim_trace_static_riro[i_shim, :], label='shimmed static + riro')
        ax.plot(shim_trace_static[i_shim, :], label='shimmed static')
        ax.plot(shim_trace_riro[i_shim, :], label='shimmed_riro')
        ax.plot(unshimmed_trace[i_shim, :], label='unshimmed')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Sum over the ROI')
        ax.legend()
        ax.set_ylim(min_value, max_value)
        ax.set_title(f"Unshimmed vs shimmed values: shim {i_shim}")
    fname_figure = os.path.join(path_output, 'fig_trace_shimmed_vs_unshimmed.png')
    fig.savefig(fname_figure)
    logger.debug(f"Saved figure: {fname_figure}")


def _print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, masked_fieldmap):
    """Print to the console metrics about the realtime and static shim. These metrics isolate temporal and static
    components
    Temporal: Compute the STD across time pixelwise, and then compute the mean across pixels.
    Static: Compute the MEAN across time pixelwise, and then compute the STD across pixels.
    """

    unshimmed_repeat = np.repeat(unshimmed[..., np.newaxis], masked_fieldmap.shape[-1], axis=-1)
    mask_repeats = np.repeat(masked_fieldmap[:, :, :, np.newaxis, :], unshimmed.shape[3], axis=3)
    ma_unshimmed = np.ma.array(unshimmed_repeat, mask=mask_repeats == False)
    ma_shim_static = np.ma.array(shimmed_static, mask=mask_repeats == False)
    ma_shim_static_riro = np.ma.array(shimmed_static_riro, mask=mask_repeats == False)
    ma_shim_riro = np.ma.array(shimmed_riro, mask=mask_repeats == False)

    # Temporal
    temp_shim_static = np.ma.mean(np.ma.std(ma_shim_static, 3))
    temp_shim_static_riro = np.ma.mean(np.ma.std(ma_shim_static_riro, 3))
    temp_shim_riro = np.ma.mean(np.ma.std(ma_shim_riro, 3))
    temp_unshimmed = np.ma.mean(np.ma.std(ma_unshimmed, 3))

    # Static
    static_shim_static = np.ma.std(np.ma.mean(ma_shim_static, 3))
    static_shim_static_riro = np.ma.std(np.ma.mean(ma_shim_static_riro, 3))
    static_shim_riro = np.ma.std(np.ma.mean(ma_shim_riro, 3))
    static_unshimmed = np.ma.std(np.ma.mean(ma_unshimmed, 3))
    logger.debug(f"\nTemporal: Compute the STD across time pixelwise, and then compute the mean across pixels."
                 f"\ntemp_shim_static: {temp_shim_static}"
                 f"\ntemp_shim_static_riro: {temp_shim_static_riro}"
                 f"\ntemp_shim_riro: {temp_shim_riro}"
                 f"\ntemp_unshimmed: {temp_unshimmed}"
                 f"\nStatic: Compute the MEAN across time pixelwise, and then compute the STD across pixels."
                 f"\nstatic_shim_static: {static_shim_static}"
                 f"\nstatic_shim_static_riro: {static_shim_static_riro}"
                 f"\nstatic_shim_riro: {static_shim_riro}"
                 f"\nstatic_unshimmed: {static_unshimmed}")


def new_bounds_from_currents(currents, old_bounds):
    """
    Uses the currents to determine the appropriate bounds for the next optimization. It assumes that
    "old_coef + next_bound < old_bound".

    Args:
        currents (np.ndarray): 2D array (n_shims x n_channels). Direct output from :func:`_optimize`.
        old_bounds (list): 1d list (n_channels) of tuples (min, max) containing the merged bounds of the previous
                           optimization.

    Returns:
        list: 2d list (n_shims x n_channels) of bounds (min, max) corresponding to each shim and channel.
    """

    new_bounds = []
    for i_shim in range(currents.shape[0]):
        shim_bound = []
        for i_channel in range(len(old_bounds)):
            a_bound = old_bounds[i_channel] - currents[i_shim, i_channel]
            shim_bound.append(tuple(a_bound))
        new_bounds.append(shim_bound)

    return new_bounds


def select_optimizer(method, unshimmed, affine, coils: ListCoil):
    """
    Select and initialize the optimizer

    Args:
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse'
        unshimmed (numpy.ndarray): 3D B0 map
        affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        coils (ListCoil): List of Coils containing the coil profiles

    Returns:
        Optimizer: Initialized Optimizer object
    """

    # global supported_optimizers
    if method in supported_optimizers:
        optimizer = supported_optimizers[method](coils, unshimmed, affine)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    return optimizer


def _optimize(optimizer: Optimizer, nii_mask_anat, slices_anat, shimwise_bounds=None,
              dilation_kernel='sphere', dilation_size=3, path_output=os.curdir):

    # Count number of channels
    n_channels = optimizer.merged_coils.shape[3]

    # Count shims to perform
    n_shims = len(slices_anat)

    # Initialize
    coefs = np.zeros((n_shims, n_channels))

    # For each shim
    for i in range(n_shims):
        # Create nibabel object of the unshimmed map
        nii_unshimmed = nib.Nifti1Image(optimizer.unshimmed, optimizer.unshimmed_affine)

        # Create mask in the fieldmap coordinate system from the anat roi mask and slice anat mask
        sliced_mask_resampled = resample_mask(nii_mask_anat, nii_unshimmed, slices_anat[i],
                                              dilation_kernel=dilation_kernel,
                                              dilation_size=dilation_size,
                                              path_output=path_output).get_fdata()

        # If new bounds are included, change them for each shim
        if shimwise_bounds is not None:
            optimizer.set_merged_bounds(shimwise_bounds[i])

        if np.all(sliced_mask_resampled == 0):
            continue

        # Optimize using the mask
        coefs[i, :] = optimizer.optimize(sliced_mask_resampled)

    return coefs


def update_affine_for_ap_slices(affine, n_slices=1, axis=2):
    """ Updates the input affine to reflect an insertion of n_slices on each side of the selected axis

    Args:
        affine (numpy.ndarray): 4x4 qform affine matrix representing the coordinates
        n_slices (int): Number of pixels to add on each side of the selected axis
        axis (int): Axis along which to insert the slice(s)

    Returns:
        (numpy.ndarray): 4x4 updated affine matrix
    """
    # Define indexes
    index_shifted = [0, 0, 0]
    index_shifted[axis] = n_slices

    # Difference of voxel in world coordinates
    spacing = apply_affine(affine, index_shifted) - apply_affine(affine, [0, 0, 0])

    # Calculate new affine
    new_affine = affine
    new_affine[:3, 3] = affine[:3, 3] - spacing

    return new_affine


def extend_slice(nii_array, n_slices=1, axis=2):
    """ Adds n_slices on each side of the selected axis. It uses the nearest slice and copies it to fill the values.
    Updates the affine of the matrix to keep the input array in the same location.

    Args:
        nii_array (nib.Nifti1Image): 3d or 4d array to extend the dimensions along an axis.
        n_slices (int): Number of slices to add on each side of the selected axis.
        axis (int): Axis along which to insert the slice(s), Allowed axis: 0, 1, 2.

    Returns:
        nib.Nifti1Image: Array extended with the appropriate affine to conserve where the original pixels were located.

    Examples:

        ::

            print(nii_array.get_fdata().shape)  # (50, 50, 1, 10)
            nii_out = extend_slice(nii_array, n_slices=1, axis=2)
            print(nii_out.get_fdata().shape)  # (50, 50, 3, 10)

    """
    if nii_array.get_fdata().ndim == 3:
        extended = nii_array.get_fdata()
        extended = extended[..., np.newaxis]
    elif nii_array.get_fdata().ndim == 4:
        extended = nii_array.get_fdata()
    else:
        raise ValueError("Unsupported number of dimensions for input array")

    for i_slice in range(n_slices):
        if axis == 0:
            extended = np.insert(extended, -1, extended[-1, :, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[0, :, :, :], axis=axis)
        elif axis == 1:
            extended = np.insert(extended, -1, extended[:, -1, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, 0, :, :], axis=axis)
        elif axis == 2:
            extended = np.insert(extended, -1, extended[:, :, -1, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, :, 0, :], axis=axis)
        else:
            raise ValueError("Unsupported value for axis")

    new_affine = update_affine_for_ap_slices(nii_array.affine, n_slices, axis)

    if nii_array.get_fdata().ndim == 3:
        extended = extended[..., 0]

    nii_extended = nib.Nifti1Image(extended, new_affine, header=nii_array.header)

    return nii_extended


def define_slices(n_slices: int, factor=1, method='sequential'):
    """
    Define the slices to shim according to the output convention. (list of tuples)

    Args:
        n_slices (int): Number of total slices.
        factor (int): Number of slices per shim.
        method (str): Defines how the slices should be sorted, supported methods include: 'interleaved', 'sequential',
                      'volume'. See Examples for more details.

    Returns:
        list: 1D list containing tuples of dim3 slices to shim. (dim1, dim2, dim3)

    Examples:

        ::

            slices = define_slices(10, 2, 'interleaved')
            print(slices)  # [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

            slices = define_slices(20, 5, 'sequential')
            print(slices)  # [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)]

            slices = define_slices(20, method='volume')
            # 'volume' ignores the 'factor' option
            print(slices)  # [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]

    """
    if n_slices <= 0:
        raise ValueError("Number of slices should be greater than 0")

    slices = []
    n_shims = n_slices // factor
    leftover = n_slices % factor

    if method == 'interleaved':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

    elif method == 'sequential':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim * factor, (i_shim + 1) * factor, 1)))

    elif method == 'volume':
        slices.append(tuple(range(n_shims)))

    else:
        raise ValueError("Not a supported method to define slices")

    if leftover != 0:
        slices.append(tuple(range(n_shims * factor, n_slices)))
        logger.warning(f"When defining the slices to shim, there are leftover slices since the factor used and number "
                       f"of slices is not perfectly dividable. Make sure the last tuple of slices is "
                       f"appropriate: {slices}")

    return slices
