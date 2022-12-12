#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression
import nibabel as nib
import logging
from nibabel.affines import apply_affine
import os
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import multiprocessing as mp
import sys

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer, PmuLsqOptimizer, allowed_opt_criteria
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.masking.mask_utils import resample_mask
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.utils import montage, timeit
from shimmingtoolbox.shim.shim_utils import calculate_metric_within_mask

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

if sys.platform == 'linux':
    mp.set_start_method('fork', force=True)
else:
    mp.set_start_method('spawn', force=True)

supported_optimizers = {
    'least_squares_rt': PmuLsqOptimizer,
    'least_squares': LsqOptimizer,
    'pseudo_inverse': Optimizer
}


@timeit
def shim_sequencer(nii_fieldmap, nii_anat, nii_mask_anat, slices, coils: ListCoil, method='least_squares',
                   opt_criteria='mse', mask_dilation_kernel='sphere', mask_dilation_kernel_size=3, reg_factor=0,
                   path_output=None):
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
        opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                            'mae': mean absolute error, 'std': standard deviation.
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.
        reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                            penalize higher current values while a lower factor will lower the effect of the
                            regularization. A negative value will favour high currents (not preferred). Only relevant
                            for 'least_squares' opt_method.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
                           artefacts.

    Returns:
        numpy.ndarray: Coefficients of the coil profiles to shim (len(slices) x n_channels)
    """

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fieldmap.get_fdata().ndim != 3:
        raise ValueError("Fieldmap must be 3d (dim1, dim2, dim3)")

    nii_fmap_orig = copy.deepcopy(nii_fieldmap)

    # Extend the fieldmap if there are axes that have less voxels than the kernel size. This is done since we are
    # fitting a fieldmap to coil profiles and having a small number of voxels can lead to errors in fitting (2 voxels
    # in one dimension can differentiate order 1 at most), the parameter allows to have at least the size of the kernel
    # for each dimension This is usually useful in the through plane direction where we could have less slices.
    # To mitigate this, we create a 3d volume by replicating the slices on the edges.
    extending = False
    for i_axis in range(3):
        if nii_fmap_orig.shape[i_axis] < mask_dilation_kernel_size:
            extending = True
            break

    if extending:
        nii_fieldmap = extend_fmap_to_kernel_size(nii_fmap_orig, mask_dilation_kernel_size, path_output)

    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim == 3:
        pass
    elif anat.ndim == 4:
        logger.info("Target anatomical is 4d, taking the average and converting to 3d")
        anat = np.mean(anat, axis=3)
        nii_anat = nib.Nifti1Image(anat, nii_anat.affine, header=nii_anat.header)
    else:
        raise ValueError("Target anatomical image must be in 3d or 4d")

    # Make sure the mask has the appropriate dimensions
    mask = nii_mask_anat.get_fdata()
    if mask.ndim == 3:
        pass
    elif mask.ndim == 4:
        logger.debug("Mask is 4d, converting to 3d")
        tmp_3d = np.zeros(mask.shape[:3])
        n_vol = mask.shape[-1]
        # Summing over 4th dimension making sure that the max value is 1
        for i_vol in range(mask.shape[-1]):
            tmp_3d += (mask[..., i_vol] / mask[..., i_vol].max())

        # 80% of the volumes must contain the desired pixel to be included, this avoids having dead voxels in the
        # output mask
        tmp_3d = threshold(tmp_3d, thr=int(n_vol * 0.8))
        nii_mask_anat = nib.Nifti1Image(tmp_3d.astype(int), nii_mask_anat.affine, header=nii_mask_anat.header)
        if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
            nib.save(nii_mask_anat, os.path.join(path_output, "fig_3d_mask.nii.gz"))
    else:
        raise ValueError("Mask must be in 3d or 4d")

    if opt_criteria not in allowed_opt_criteria:
        raise ValueError("Criteria for optimization not supported")

    # Resample the input mask on the target anatomical image if they are different
    if not np.all(nii_mask_anat.shape == anat.shape) or not np.all(nii_mask_anat.affine == nii_anat.affine):
        logger.debug("Resampling mask on the target anat")
        nii_mask_anat_soft = resample_from_to(nii_mask_anat, nii_anat, order=1, mode='grid-constant')
        tmp_mask = nii_mask_anat_soft.get_fdata()
        # Change soft mask into binary mask
        tmp_mask = threshold(tmp_mask, thr=0.001)
        nii_mask_anat = nib.Nifti1Image(tmp_mask, nii_mask_anat_soft.affine, header=nii_mask_anat_soft.header)

        if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
            nib.save(nii_mask_anat, os.path.join(path_output, "mask_static_resampled_on_anat.nii.gz"))

    # Select and initialize the optimizer
    optimizer = select_optimizer(method, fieldmap, affine_fieldmap, coils, opt_criteria, reg_factor=reg_factor)

    # Optimize slice by slice
    logger.info("Optimizing")
    coefs = _optimize(optimizer, nii_mask_anat, slices, dilation_kernel=mask_dilation_kernel,
                      dilation_size=mask_dilation_kernel_size, path_output=path_output)

    # Evaluate theoretical shim
    logger.info("Calculating output files and preparing figures")
    _eval_static_shim(optimizer, nii_fmap_orig, nii_mask_anat, coefs, slices, path_output)

    return coefs


@timeit
def _eval_static_shim(opt: Optimizer, nii_fieldmap_orig, nii_mask, coef, slices, path_output):
    """Calculate theoretical shimmed map and output figures"""

    # Save the merged coil profiles if in debug
    if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
        # Save coils
        nii_merged_coils = nib.Nifti1Image(opt.merged_coils, nii_fieldmap_orig.affine, header=nii_fieldmap_orig.header)
        nib.save(nii_merged_coils, os.path.join(path_output, "merged_coils.nii.gz"))

    unshimmed = nii_fieldmap_orig.get_fdata()

    # If the fieldmap was changed (i.e. only 1 slice) we want to evaluate the output on the original fieldmap
    merged_coils, _ = opt.merge_coils(unshimmed, nii_fieldmap_orig.affine)

    # Initialize
    shimmed = np.zeros(unshimmed.shape + (len(slices),))
    corrections = np.zeros(unshimmed.shape + (len(slices),))
    masks_fmap = np.zeros(unshimmed.shape + (len(slices),))
    for i_shim in range(len(slices)):
        # Calculate shimmed values
        correction_per_channel = coef[i_shim] * merged_coils
        corrections[..., i_shim] = np.sum(correction_per_channel, axis=3, keepdims=False)
        shimmed[..., i_shim] = unshimmed + corrections[..., i_shim]

        # Create non binary mask
        masks_fmap[..., i_shim] = resample_mask(nii_mask, nii_fieldmap_orig, slices[i_shim]).get_fdata()

        ma_shimmed = np.ma.array(shimmed[..., i_shim], mask=masks_fmap[..., i_shim]==False)
        ma_unshimmed = np.ma.array(unshimmed, mask=masks_fmap[..., i_shim]==False)
        std_shimmed = np.ma.std(ma_shimmed)
        std_unshimmed = np.ma.std(ma_unshimmed)
        mae_shimmed = np.ma.mean(np.ma.abs(ma_shimmed))
        mae_unshimmed = np.ma.mean(np.ma.abs(ma_unshimmed))
        mse_shimmed = np.ma.mean(np.square(ma_shimmed))
        mse_unshimmed = np.ma.mean(np.square(ma_unshimmed))

        if mse_shimmed > mse_unshimmed:
            logger.warning("Verify the shim parameters. Some give worse results than no shim.\n"
                           f"i_shim: {i_shim}")

        logger.debug(f"Slice(s): {slices[i_shim]}\n"
                     f"MAE:\n"
                     f"unshimmed: {mae_unshimmed}, shimmed: {mae_shimmed}\n"
                     f"MSE:\n"
                     f"unshimmed: {mse_unshimmed}, shimmed: {mse_shimmed}\n"
                     f"STD:\n"
                     f"unshimmed: {std_unshimmed}, shimmed: {std_shimmed}"
                     f"current: \n{coef[i_shim, :]}")

    # Figure that shows unshimmed vs shimmed for each slice
    if path_output is not None:
        # fmap space
        # Merge the i_shim into one single fieldmap shimmed (correction applied only where it will be applied on the
        # fieldmap)
        shimmed_masked, mask_full_binary = _calc_shimmed_full_mask(unshimmed, corrections, nii_mask, nii_fieldmap_orig,
                                                                   slices, masks_fmap)

        if len(slices) == 1:
            # TODO: Output json sidecar
            # TODO: Update the shim settings if Scanner coil?
            # Output the resulting fieldmap since it can be calculated over the entire fieldmap
            nii_shimmed_fmap = nib.Nifti1Image(shimmed[..., 0], nii_fieldmap_orig.affine,
                                               header=nii_fieldmap_orig.header)
            fname_shimmed_fmap = os.path.join(path_output, 'fieldmap_calculated_shim.nii.gz')
            nib.save(nii_shimmed_fmap, fname_shimmed_fmap)
        else:
            # Output the resulting masked fieldmap since it cannot be calculated over the entire fieldmap
            nii_shimmed_fmap = nib.Nifti1Image(shimmed_masked, nii_fieldmap_orig.affine,
                                               header=nii_fieldmap_orig.header)
            fname_shimmed_fmap = os.path.join(path_output, 'fieldmap_calculated_shim_masked.nii.gz')
            nib.save(nii_shimmed_fmap, fname_shimmed_fmap)

        # Save images to a file
        # TODO: Add units if possible
        # TODO: Add in anat space?
        _plot_static_full_mask(unshimmed, shimmed_masked, mask_full_binary, path_output)
        _plot_static_partial_mask(unshimmed, shimmed, masks_fmap, path_output)
        _plot_currents(coef, path_output)
        _cal_shimmed_anat_orient(coef, merged_coils, nii_mask, nii_fieldmap_orig, slices, path_output)

        if logger.level <= getattr(logging, 'DEBUG'):
            # Save to a NIfTI
            fname_correction = os.path.join(path_output, 'fig_correction.nii.gz')
            nii_correction_3d = nib.Nifti1Image(shimmed_masked, opt.unshimmed_affine)
            nib.save(nii_correction_3d, fname_correction)

            # 4th dimension is i_shim
            fname_correction = os.path.join(path_output, 'fig_shimmed_4thdim_ishim.nii.gz')
            nii_correction = nib.Nifti1Image(masks_fmap * shimmed, opt.unshimmed_affine)
            nib.save(nii_correction, fname_correction)


def _cal_shimmed_anat_orient(coefs, coils, nii_mask_anat, nii_fieldmap, slices, path_output):
    nii_coils = nib.Nifti1Image(coils, nii_fieldmap.affine, header=nii_fieldmap.header)
    coils_anat = resample_from_to(nii_coils,
                                  nii_mask_anat,
                                  order=1,
                                  mode='grid-constant',
                                  cval=0).get_fdata()
    fieldmap_anat = resample_from_to(nii_fieldmap,
                                     nii_mask_anat,
                                     order=1,
                                     mode='grid-constant',
                                     cval=0).get_fdata()

    shimmed_anat_orient = np.zeros_like(fieldmap_anat)
    for i_shim in range(len(slices)):
        corr = np.sum(coefs[i_shim] * coils_anat, axis=3, keepdims=False)
        shimmed_anat_orient[..., slices[i_shim]] = fieldmap_anat[..., slices[i_shim]] + corr[..., slices[i_shim]]

    fname_shimmed_anat_orient = os.path.join(path_output, 'fig_shimmed_anat_orient.nii.gz')
    nii_shimmed_anat_orient = nib.Nifti1Image(shimmed_anat_orient * nii_mask_anat.get_fdata(), nii_mask_anat.affine,
                                              header=nii_mask_anat.header)
    nib.save(nii_shimmed_anat_orient, fname_shimmed_anat_orient)


def _calc_shimmed_full_mask(unshimmed, correction, nii_mask_anat, nii_fieldmap, slices, masks_fmap):
    mask_full_binary = np.clip(np.ceil(resample_from_to(nii_mask_anat,
                                                        nii_fieldmap,
                                                        order=0,
                                                        mode='grid-constant',
                                                        cval=0).get_fdata()), 0, 1)

    # Find the correction
    full_correction = np.zeros(unshimmed.shape)
    for i_shim in range(len(slices)):
        # Apply the correction weighted according to the mask
        full_correction += correction[..., i_shim] * masks_fmap[..., i_shim]

    # Calculate the weighted whole mask
    mask_weight = np.sum(masks_fmap, axis=3)
    # Divide by the weighted mask. This is done so that the edges of the soft mask can be shimmed appropriately
    full_correction_scaled = np.divide(full_correction, mask_weight, where=mask_full_binary.astype(bool))

    # Apply the correction to the unshimmed image
    shimmed_masked = (full_correction_scaled + unshimmed) * mask_full_binary

    return shimmed_masked, mask_full_binary


def _plot_static_partial_mask(unshimmed, shimmed, masks, path_output):
    a_slice = 0
    unshimmed_repeated = np.repeat(unshimmed[..., np.newaxis], masks.shape[-1], axis=3)
    mt_unshimmed = montage(unshimmed_repeated[:, :, a_slice, :])
    mt_shimmed = montage(shimmed[:, :, a_slice, :])
    unshimmed_masked_repeated = np.repeat(unshimmed[..., np.newaxis], masks.shape[-1], axis=3) * np.ceil(masks)
    mt_unshimmed_masked = montage(unshimmed_masked_repeated[:, :, a_slice, :])
    mt_shimmed_masked = montage(shimmed[:, :, a_slice, :] * np.ceil(masks[:, :, a_slice, :]))

    min_masked_value = min(mt_unshimmed_masked.min(), mt_shimmed_masked.min())
    max_masked_value = max(mt_unshimmed_masked.max(), mt_shimmed_masked.max())

    min_fmap_value = min(mt_unshimmed.min(), mt_shimmed.min())
    max_fmap_value = max(mt_unshimmed.max(), mt_shimmed.max())

    fig = Figure(figsize=(8, 5))
    fig.suptitle(f"Fieldmaps for all shim groups\nFieldmap Coordinate System")

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(mt_unshimmed, vmin=min_fmap_value, vmax=max_fmap_value, cmap='gray')
    mt_unshimmed_masked[mt_unshimmed_masked == 0] = np.nan
    im = ax.imshow(mt_unshimmed_masked, vmin=min_masked_value, vmax=max_masked_value, cmap='viridis')
    ax.set_title("Unshimmed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mt_shimmed, vmin=min_fmap_value, vmax=max_fmap_value, cmap='gray')
    mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
    im = ax.imshow(mt_shimmed_masked, vmin=min_masked_value, vmax=max_masked_value, cmap='viridis')
    ax.set_title("Shimmed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    # Save
    fname_figure = os.path.join(path_output, 'fig_shimmed_vs_unshimmed_shim_groups.png')
    fig.savefig(fname_figure, bbox_inches='tight')


def _plot_static_full_mask(unshimmed, shimmed_masked, mask, path_output):
    # Plot
    mt_unshimmed = montage(unshimmed)
    mt_unshimmed_masked = montage(unshimmed * mask)
    mt_shimmed_masked = montage(shimmed_masked)

    metric_unshimmed_std = calculate_metric_within_mask(unshimmed, mask, metric='std')
    metric_shimmed_std = calculate_metric_within_mask(shimmed_masked, mask, metric='std')
    metric_unshimmed_mean = calculate_metric_within_mask(unshimmed, mask, metric='mean')
    metric_shimmed_mean = calculate_metric_within_mask(shimmed_masked, mask, metric='mean')
    metric_unshimmed_absmean = calculate_metric_within_mask(np.abs(unshimmed), mask, metric='mean')
    metric_shimmed_absmean = calculate_metric_within_mask(np.abs(shimmed_masked), mask, metric='mean')

    min_value = min(mt_unshimmed_masked.min(), mt_shimmed_masked.min())
    max_value = max(mt_unshimmed_masked.max(), mt_shimmed_masked.max())

    fig = Figure(figsize=(9, 6))
    fig.suptitle(f"Fieldmaps\nFieldmap Coordinate System")

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(mt_unshimmed, cmap='gray')
    mt_unshimmed_masked[mt_unshimmed_masked == 0] = np.nan
    im = ax.imshow(mt_unshimmed_masked, vmin=min_value, vmax=max_value, cmap='viridis')
    ax.set_title(f"Before shimming\nSTD: {metric_unshimmed_std:.3}, mean: {metric_unshimmed_mean:.3}, "
                 f"abs mean: {metric_unshimmed_absmean:.3}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mt_unshimmed, cmap='gray')
    mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
    im = ax.imshow(mt_shimmed_masked, vmin=min_value, vmax=max_value, cmap='viridis')
    ax.set_title(f"After shimming\nSTD: {metric_shimmed_std:.3}, mean: {metric_shimmed_mean:.3}, "
                 f"abs mean: {metric_shimmed_absmean:.3}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    # Lower suptitle
    fig.subplots_adjust(top=0.85)

    # Save
    fname_figure = os.path.join(path_output, 'fig_shimmed_vs_unshimmed.png')
    fig.savefig(fname_figure, bbox_inches='tight')


@timeit
def shim_realtime_pmu_sequencer(nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices,
                                pmu: PmuResp, coils: ListCoil, opt_method='least_squares', opt_criteria='mse',
                                reg_factor=0, mask_dilation_kernel='sphere', mask_dilation_kernel_size=3,
                                path_output=None):
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
        opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                            'mae': mean absolute error, 'std': standard deviation.
        reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                            penalize higher current values while a lower factor will lower the effect of the
                            regularization. A negative value will favour high currents (not preferred). Only relevant
                            for 'least_squares' opt_method.
        mask_dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
                           artefacts.

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

    nii_fmap_orig = copy.deepcopy(nii_fieldmap)
    # Extend the fieldmap if there are axes that have less voxels than the kernel size. This is done since we are
    # fitting a fieldmap to coil profiles and having a small number of voxels can lead to errors in fitting (2 voxels
    # in one dimension can differentiate order 1 at most), the parameter allows to have at least the size of the kernel
    # for each dimension This is usually useful in the through plane direction where we could have less slices.
    # To mitigate this, we create a 3d volume by replicating the slices on the edges.
    extending = False
    for i_axis in range(3):
        if nii_fmap_orig.shape[i_axis] < mask_dilation_kernel_size:
            extending = True
            break

    if extending:
        nii_fieldmap = extend_fmap_to_kernel_size(nii_fmap_orig, mask_dilation_kernel_size, path_output)

    fieldmap = nii_fieldmap.get_fdata()
    affine_fieldmap = nii_fieldmap.affine

    # Make sure anat has the appropriate dimensions
    anat = nii_anat.get_fdata()
    if anat.ndim != 3:
        raise ValueError("Anatomical image must be in 3d")

    # Make sure masks have the appropriate dimensions
    if nii_static_mask.get_fdata().ndim != 3:
        raise ValueError("static_mask image must be in 3d")
    if nii_riro_mask.get_fdata().ndim != 3:
        raise ValueError("riro_mask image must be in 3d")

    if opt_criteria not in allowed_opt_criteria:
        raise ValueError("Criteria for optimization not supported")

    # Resample the input masks on the target anatomical image if they are different
    if not np.all(nii_static_mask.shape == anat.shape) or not np.all(nii_static_mask.affine == nii_anat.affine):
        logger.debug("Resampling static mask on the target anat")
        nii_static_mask_soft = resample_from_to(nii_static_mask, nii_anat, order=1, mode='grid-constant')
        tmp_mask = nii_static_mask_soft.get_fdata()
        # Change soft mask into binary mask
        tmp_mask = threshold(tmp_mask, thr=0.001)
        nii_static_mask = nib.Nifti1Image(tmp_mask, nii_static_mask_soft.affine, header=nii_static_mask_soft.header)

        if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
            nib.save(nii_static_mask, os.path.join(path_output, "mask_static_resampled_on_anat.nii.gz"))

    if not np.all(nii_riro_mask.shape == anat.shape) or not np.all(nii_riro_mask.affine == nii_anat.affine):
        logger.debug("Resampling riro mask on the target anat")
        nii_riro_mask_soft = resample_from_to(nii_riro_mask, nii_anat, order=1, mode='grid-constant')
        tmp_mask = nii_riro_mask_soft.get_fdata()
        # Change soft mask into binary mask
        tmp_mask = threshold(tmp_mask, thr=0.001)
        nii_riro_mask = nib.Nifti1Image(tmp_mask, nii_riro_mask_soft.affine, header=nii_riro_mask_soft.header)

        if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
            nib.save(nii_riro_mask, os.path.join(path_output, "mask_riro_resampled_on_anat.nii.gz"))

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

    # Safety check for linear regression if the pressure and fieldmap fit well
    # Mask the voxels not being shimmed for riro
    nii_3dfmap = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine, header=nii_fieldmap.header)
    fmap_mask_riro = resample_mask(nii_riro_mask, nii_3dfmap, tuple(range(anat.shape[2])),
                                   dilation_kernel=mask_dilation_kernel,
                                   dilation_size=mask_dilation_kernel_size).get_fdata()
    masked_fieldmap_riro = np.repeat(fmap_mask_riro[..., np.newaxis], fieldmap.shape[-1], 3) * fieldmap
    y = masked_fieldmap_riro.reshape(-1, fieldmap.shape[-1]).T

    reg_riro = LinearRegression().fit(x, y)
    # Calculate adjusted r2 score (Takes into account the number of observations and predictor variables)
    score_riro = 1 - (1 - reg_riro.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
    logger.debug(f"Linear fit of the RIRO masked fieldmap and pressure got a R2 score of: {score_riro}")

    # Warn if lower than a threshold
    # Threshold was set by looking at a small sample of data (This value could be updated based on user feedback)
    threshold_score = 0.7
    if score_riro < threshold_score:
        logger.warning(f"Linear fit of the RIRO masked fieldmap and pressure got a low R2 score: {score_riro} "
                       f"(less than {threshold_score}). This indicates a bad fit between the pressure data and the "
                       f"fieldmap values")

    # Fit to the linear model (no mask)
    y = fieldmap.reshape(-1, fieldmap.shape[-1]).T
    reg = LinearRegression().fit(x, y)

    # static/riro contains a 3d matrix of static/riro map in the fieldmap space considering the previous equation
    static = reg.intercept_.reshape(fieldmap.shape[:-1])
    riro = reg.coef_.reshape(fieldmap.shape[:-1])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

    # Log the static and riro maps to fit
    if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
        # Save static
        nii_static = nib.Nifti1Image(static, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_static, os.path.join(path_output, 'fig_static_fmap_component.nii.gz'))

        # Save riro
        nii_riro = nib.Nifti1Image(riro, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_riro, os.path.join(path_output, 'fig_riro_fmap_component.nii.gz'))

    # Static shim
    optimizer = select_optimizer(opt_method, static, affine_fieldmap, coils, opt_criteria, reg_factor=reg_factor)
    logger.info("Static optimization")
    coef_static = _optimize(optimizer, nii_static_mask, slices,
                            dilation_kernel=mask_dilation_kernel,
                            dilation_size=mask_dilation_kernel_size,
                            path_output=path_output)

    # RIRO optimization
    # Use the currents to define a list of new coil bounds for the riro optimization
    bounds = new_bounds_from_currents(coef_static, optimizer.merged_bounds)

    if opt_method == 'least_squares':
        opt_method = 'least_squares_rt'

    optimizer = select_optimizer(opt_method, riro, affine_fieldmap, coils, opt_criteria, pmu, reg_factor=reg_factor)
    logger.info("Realtime optimization")
    coef_riro = _optimize(optimizer, nii_riro_mask, slices,
                          shimwise_bounds=bounds,
                          dilation_kernel=mask_dilation_kernel,
                          dilation_size=mask_dilation_kernel_size,
                          path_output=path_output)

    # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
    # between scans. This allows to compare results between scans.
    # coef_riro_rms = coef_riro * pressure_rms
    # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

    # Evaluate theoretical shim
    logger.info("Calculating output files and preparing figures")
    _eval_rt_shim(optimizer, nii_fieldmap, nii_static_mask, coef_static, coef_riro, mean_p,
                  acq_pressures, slices, pressure_rms, pmu, path_output)

    return coef_static, coef_riro, mean_p, pressure_rms


@timeit
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
    mask_fmap_cs = np.zeros(unshimmed[..., 0].shape + (len(slices),))
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(coef_static[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        # Calculate the riro coil profiles
        riro_profile = np.sum(coef_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        mask_fmap_cs[..., i_shim] = np.ceil(resample_mask(nii_mask_static, nii_target, slices[i_shim]).get_fdata())
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
            shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
            shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

            # Calculate masked shim
            masked_shim_static[..., i_t, i_shim] = mask_fmap_cs[..., i_shim] * shimmed_static[..., i_t, i_shim]
            masked_shim_static_riro[..., i_t, i_shim] = mask_fmap_cs[..., i_shim] * shimmed_static_riro[..., i_t,
                                                                                                        i_shim]
            masked_shim_riro[..., i_t, i_shim] = mask_fmap_cs[..., i_shim] * shimmed_riro[..., i_t, i_shim]
            masked_unshimmed[..., i_t, i_shim] = mask_fmap_cs[..., i_shim] * unshimmed[..., i_t]

            # Calculate the sum over the ROI
            # TODO: Calculate the sum of mask_fmap_cs[..., i_shim] and divide by that (If bigger roi due to
            #  interpolation, it should not count more) Psibly use soft mask?
            sum_shimmed_static = np.sum(np.abs(masked_shim_static[..., i_t, i_shim]))
            sum_shimmed_static_riro = np.sum(np.abs(masked_shim_static_riro[..., i_t, i_shim]))
            sum_shimmed_riro = np.sum(np.abs(masked_shim_riro[..., i_t, i_shim]))
            sum_unshimmed = np.sum(np.abs(masked_unshimmed[..., i_t, i_shim]))

            if sum_shimmed_static_riro > sum_unshimmed:
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

    if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
        _plot_static_riro(masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed, shimmed_static,
                          shimmed_static_riro, path_output, slices, i_slice=i_slice, i_shim=i_shim, i_t=i_t)
        _plot_currents(coef_static, path_output, riro=coef_riro * pressure_rms)
        _plot_shimmed_trace(unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro,
                            path_output)
        _plot_pressure_points(acq_pressures, (pmu.min, pmu.max), path_output)
        _print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, mask_fmap_cs)

        # Save shimmed result
        nii_shimmed_static_riro = nib.Nifti1Image(shimmed_static_riro, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_shimmed_static_riro, os.path.join(path_output,
                                                       'shimmed_static_riro_4thdim_it_5thdim_ishim.nii.gz'))

        # Save coils
        nii_merged_coils = nib.Nifti1Image(opt.merged_coils, nii_fieldmap.affine, header=nii_fieldmap.header)
        nib.save(nii_merged_coils, os.path.join(path_output, "merged_coils.nii.gz"))


def _plot_static_riro(masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed, shimmed_static,
                      shimmed_static_riro, path_output, slices, i_t=0, i_slice=0, i_shim=0):
    """Plot Static and RIRO fieldmap for a perticular fieldmap slice, anat shim and timepoint"""

    min_value = min(masked_shim_static_riro[..., i_slice, i_t, i_shim].min(),
                    masked_shim_static[..., i_slice, i_t, i_shim].min(),
                    masked_unshimmed[..., i_slice, i_t, i_shim].min())
    max_value = max(masked_shim_static_riro[..., i_slice, i_t, i_shim].max(),
                    masked_shim_static[..., i_slice, i_t, i_shim].max(),
                    masked_unshimmed[..., i_slice, i_t, i_shim].max())

    index_slice_to_show = slices[i_shim][i_slice]

    fig = Figure(figsize=(15, 10))
    fig.suptitle(f"Maps for slice: {index_slice_to_show}, timepoint: {i_t}")
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
        ax.plot(static[:, i_channel], label=f"Static channel{i_channel} currents through shim groups")
    if riro is not None:
        for i_channel in range(n_channels):
            ax.plot(riro[:, i_channel], label=f"Riro channel{i_channel} currents through shim groups")
    ax.set_xlabel('Shim group')
    ax.set_ylabel('Coefficients (Physical CS [RAS])')
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


def _print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, mask):
    """Print to the console metrics about the realtime and static shim. These metrics isolate temporal and static
    components
    Temporal: Compute the STD across time pixelwise, and then compute the mean across pixels.
    Static: Compute the MEAN across time pixelwise, and then compute the STD across pixels.
    """

    unshimmed_repeat = np.repeat(unshimmed[..., np.newaxis], mask.shape[-1], axis=-1)
    mask_repeats = np.repeat(mask[:, :, :, np.newaxis, :], unshimmed.shape[3], axis=3)
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
        list: 2d list (n_shim_groups x n_channels) of bounds (min, max) corresponding to each shim group and channel.
    """

    new_bounds = []
    for i_shim in range(currents.shape[0]):
        shim_bound = []
        for i_channel in range(len(old_bounds)):
            a_bound = old_bounds[i_channel] - currents[i_shim, i_channel]
            shim_bound.append(tuple(a_bound))
        new_bounds.append(shim_bound)

    return new_bounds


def select_optimizer(method, unshimmed, affine, coils: ListCoil, opt_criteria, pmu: PmuResp = None, reg_factor=0):
    """
    Select and initialize the optimizer

    Args:
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'least_squares_rt'
        unshimmed (numpy.ndarray): 3D B0 map
        affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        coils (ListCoil): List of Coils containing the coil profiles
        opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                            'mae': mean absolute error, 'std': standard deviation.
        pmu (PmuResp): PmuResp object containing the respiratory trace information. Required for method
                       'least_squares_rt'.
        reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                    penalize higher current values while a lower factor will lower the effect of the
                    regularization. A negative value will favour high currents (not preferred).

    Returns:
        Optimizer: Initialized Optimizer object
    """

    # global supported_optimizers
    if method in supported_optimizers:
        if method == 'least_squares':
            optimizer = supported_optimizers[method](coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)

        elif method == 'least_squares_rt':
            # Make sure pmu is defined
            if pmu is None:
                raise ValueError(f"pmu parameter is required if using the optimization method: {method}")

            # Add pmu to the realtime optimizer(s)
            optimizer = supported_optimizers[method](coils, unshimmed, affine, opt_criteria, pmu, reg_factor=reg_factor)
        else:
            optimizer = supported_optimizers[method](coils, unshimmed, affine)
    else:
        raise KeyError(f"Method: {method} is not part of the supported optimizers")

    return optimizer


def _optimize(optimizer: Optimizer, nii_mask_anat, slices_anat, shimwise_bounds=None,
              dilation_kernel='sphere', dilation_size=3, path_output=None):
    # Count shims to perform
    n_shims = len(slices_anat)

    # multiprocessing optimization
    _optimize_scope = (
        optimizer, nii_mask_anat, slices_anat, dilation_kernel, dilation_size, path_output, shimwise_bounds)

    # Default number of workers is set to mp.cpu_count()
    # _worker_init gets called by each worker with _optimize_scope as arguments
    # _worker_init converts those arguments as globals so they can be accessed in _opt
    # This works because each worker has its own version of the global variables
    # This allows to use both fork and spawn while not serializing the arguments making it slow
    # It also allows to give as input only 1 iterable (range(n_shims))) so 'starmap' does not have to be used

    # 'imap_unordered' is used since a worker returns the value when it is done instead of waiting for the whole call
    # to 'map', 'starmap' to finish. This allows to show progress. 'imap' is similar to 'imap_unordered' but since it
    # returns in order, the progress is less accurate. Even though 'map_async' and 'starmap_async' do not block, the
    # whole call needs to be finished to access the results (results.get()).
    # A whole discussion thread is available here:
    # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    pool = mp.Pool(initializer=_worker_init, initargs=_optimize_scope)
    try:

        results = []
        print(f"\rProgress 0.0%")
        for i, result in enumerate(pool.imap_unordered(_opt, range(n_shims))):
            print(f"\rProgress {np.round((i + 1)/n_shims * 100)}%")
            results.append(result)

    except mp.context.TimeoutError:
        logger.info("Multiprocessing might have hung, retry the same command")
    finally:
        pool.close()
        pool.join()

    results.sort(key=lambda x: x[0])
    results_final = [r for i, r in results]

    return np.array(results_final)


gl_optimizer = None
gl_nii_mask_anat = None
gl_slices_anat = None
gl_dilation_kernel = None
gl_dilation_size = None
gl_path_output = None
gl_shimwise_bounds = None


def _worker_init(optimizer, nii_mask_anat, slices_anat, dilation_kernel, dilation_size, path_output,
                 shimwise_bounds):
    global gl_optimizer, gl_nii_mask_anat, gl_slices_anat, gl_dilation_kernel
    global gl_dilation_size, gl_path_output, gl_shimwise_bounds
    gl_optimizer = optimizer
    gl_nii_mask_anat = nii_mask_anat
    gl_slices_anat = slices_anat
    gl_dilation_kernel = dilation_kernel
    gl_dilation_size = dilation_size
    gl_path_output = path_output
    gl_shimwise_bounds = shimwise_bounds


def _opt(i):

    # Create nibabel object of the unshimmed map
    nii_unshimmed = nib.Nifti1Image(gl_optimizer.unshimmed, gl_optimizer.unshimmed_affine)

    # Create mask in the fieldmap coordinate system from the anat roi mask and slice anat mask
    sliced_mask_resampled = resample_mask(gl_nii_mask_anat, nii_unshimmed, gl_slices_anat[i],
                                          dilation_kernel=gl_dilation_kernel,
                                          dilation_size=gl_dilation_size,
                                          path_output=gl_path_output).get_fdata()

    # If new bounds are included, change them for each shim
    if gl_shimwise_bounds is not None:
        gl_optimizer.set_merged_bounds(gl_shimwise_bounds[i])

    if np.all(sliced_mask_resampled == 0):
        return i, np.zeros(gl_optimizer.merged_coils.shape[-1])

    # Optimize using the mask
    coef = gl_optimizer.optimize(sliced_mask_resampled)

    return i, coef


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


def extend_fmap_to_kernel_size(nii_fmap_orig, dilation_kernel_size, path_output=None):
    """ Load the fmap and expand its dimensions to the kernel size

    Args:
        nii_fmap_orig (nib.Nifti1Image): 3d (dim1, dim2, dim3) or 4d (dim1, dim2, dim3, t) nii to be extended
        dilation_kernel_size: Size of the kernel
        path_output (str): Path to save the debug output

    Returns:
        nibabel.Nifti1Image: Nibabel object of the loaded and extended fieldmap

    """

    fieldmap_shape = nii_fmap_orig.shape[:3]

    # Extend the dimensions where the kernel is bigger than the number of voxels
    tmp_nii = copy.deepcopy(nii_fmap_orig)
    for i_axis in range(len(fieldmap_shape)):
        # If there are less voxels than the kernel size, extend in that axis
        if fieldmap_shape[i_axis] < dilation_kernel_size:
            diff = float(dilation_kernel_size - fieldmap_shape[i_axis])
            n_slices_to_extend = math.ceil(diff / 2)
            tmp_nii = extend_slice(tmp_nii, n_slices=n_slices_to_extend, axis=i_axis)

    nii_fmap = tmp_nii

    # If DEBUG, save the extended fieldmap
    if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
        fname_new_fmap = os.path.join(path_output, 'tmp_extended_fmap.nii.gz')
        nib.save(nii_fmap, fname_new_fmap)
        logger.debug(f"Extended fmap, saved the new fieldmap here: {fname_new_fmap}")

    return nii_fmap


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


def parse_slices(fname_nifti):
    """ Parse the BIDS sidecar associated with the input nifti file.

    Args:
        fname_nifti (str): Full path to a NIfTI file

    Returns:
        list: 1D list containing tuples of dim3 slices to shim. (dim1, dim2, dim3)
    """

    # Open json
    fname_json = fname_nifti.split('.nii')[0] + '.json'
    # Read from json file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # The BIDS specification mentions that the 'SliceTiming' is stored on disk depending on the
    # 'SliceEncodingDirection'. If this tag is 'i', 'j', 'k' or non existent, index 0 of 'SliceTiming' corresponds to
    # index 0 of the slice dimension of the NIfTI file. If 'SliceEncodingDirection' is 'i-', 'j-' or 'k-',
    # the last value of 'SliceTiming' corresponds to index 0 of the slice dimension of the NIfTI file.
    # https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#timing-parameters

    # Note: Dcm2niix does not seem to include the tag 'SliceEncodingDirection' and always makes sure index 0 of
    # 'SliceTiming' corresponds to index 0 of the NIfTI file.
    # https://www.nitrc.org/forum/forum.php?thread_id=10307&forum_id=4703
    # https://github.com/rordenlab/dcm2niix/issues/530

    # Make sure tag SliceTiming exists
    if 'SliceTiming' in json_data:
        slice_timing = json_data['SliceTiming']
    else:
        raise RuntimeError("No tag SliceTiming to parse slice data")

    # If SliceEncodingDirection exists and is negative, SliceTiming is reversed
    if 'SliceEncodingDirection' in json_data:
        if json_data['SliceEncodingDirection'][-1] == '-':
            logger.debug("SliceEncodeDirection is negative, SliceTiming parsed backwards")
            slice_timing.reverse()

    # Return the indexes of the sorted slice_timing
    slice_timing = np.array(slice_timing)
    list_slices = np.argsort(slice_timing)
    slices = []
    # Construct the list of tuples
    while len(list_slices) > 0:
        # Find if the first index has the same timing as other indexes
        # shim_group = tuple(list_slices[list_slices == list_slices[0]])
        shim_group = tuple(np.where(slice_timing == slice_timing[list_slices[0]])[0])
        # Add this as a tuple
        slices.append(shim_group)

        # Since the list_slices is sorted by slice_timing, the only similar values will be at the beginning
        n_to_remove = len(shim_group)
        list_slices = list_slices[n_to_remove:]

    return slices


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
    leftover = 0

    if method == 'interleaved':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

        leftover = n_slices % factor

    elif method == 'sequential':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim * factor, (i_shim + 1) * factor, 1)))

        leftover = n_slices % factor

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


def shim_max_intensity(nii_input, nii_mask=None):
    """ Find indexes of the 4th dimension of the input volume that has the highest signal intensity for each slice.
        Based on: https://onlinelibrary.wiley.com/doi/10.1002/hbm.26018

    Args:
        nii_input (nib.Nifti1Image): 4d volume where 4th dimension was acquired with different shim values
        nii_mask (nib.Nifti1Image): Mask defining the spatial region to shim. If None: consider all voxels of nii_input.

    Returns:
        np.ndarray: 1d array containing the index of the volume that maximizes signal intensity for each slice

    """

    if len(nii_input.shape) != 4:
        raise ValueError("Input volume must be 4d")

    # Load the mask
    if nii_mask is None:
        mask = np.ones(nii_input.shape[:3])
    else:
        # Masks must be 3d
        if len(nii_mask.shape) != 3:
            raise ValueError("Input mask must be 3d")
        # If the mask is of a different shape, resample it.
        elif not np.all(nii_mask.shape == nii_input.shape[:3]) or not np.all(nii_mask.affine == nii_input.affine):
            nii_input_3d = nib.Nifti1Image(nii_input.get_fdata()[..., 0], nii_input.affine, header=nii_input.header)
            mask = resample_mask(nii_mask, nii_input_3d).get_fdata()
        else:
            mask = nii_mask.get_fdata()

    n_slices = nii_input.shape[2]
    n_volumes = nii_input.shape[3]

    mean_values = np.zeros([n_slices, n_volumes])
    for i_volume in range(n_volumes):
        masked_epi_3d = nii_input.get_fdata()[..., i_volume] * mask
        mean_per_slice = np.mean(masked_epi_3d, axis=(0, 1), where=mask.astype(bool))
        mean_values[:, i_volume] = mean_per_slice

    if np.any(np.isnan(mean_values)):
        logger.warning("NaN values when calculating the mean. This is usually because the mask is not defined in all "
                       "slices. The output will disregard slices with NaN values.")

    index_per_slice = np.nanargmax(mean_values, axis=1)

    return index_per_slice
