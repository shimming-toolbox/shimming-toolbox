#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
import math
import numpy as np
from joblib import delayed, Parallel
from typing import List
from sklearn.linear_model import LinearRegression
import nibabel as nib
import logging
from nibabel.affines import apply_affine
import os
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer, PmuLsqOptimizer, allowed_opt_criteria
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.optimizer.quadprog_optimizer import QuadProgOpt, PmuQuadProgOpt
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.masking.mask_utils import resample_mask
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.utils import montage
from shimmingtoolbox.shim.shim_utils import calculate_metric_within_mask

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

supported_optimizers = {
    'least_squares_rt': PmuLsqOptimizer,
    'least_squares': LsqOptimizer,
    'quad_prog': QuadProgOpt,
    'quad_prog_rt': PmuQuadProgOpt,
    'pseudo_inverse': Optimizer
}


class Sequencer(object):
    """
    General class for the sequencer

    Attributes:
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape
                       of anat is: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                         For example, a kernel of size 3 will dilate the mask by 1 pixel.
        reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                            penalize higher current values while a lower factor will lower the effect of the
                            regularization. A negative value will favour high currents (not preferred). Only relevant
                            for 'least_squares' opt_method.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
    """

    def __init__(self, slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor, path_output):
        """
        Constructor of the sequencer class

        Args:
            slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape
                           of anat is: (dim1, dim2, dim3). Refer to
                           :func:`shimmingtoolbox.shim.sequencer.define_slices`.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for
                                        more details.
            mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                             For example, a kernel of size 3 will dilate the mask by 1 pixel.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
        """
        self.slices = slices
        self.mask_dilation_kernel = mask_dilation_kernel
        self.mask_dilation_kernel_size = mask_dilation_kernel_size
        self.reg_factor = reg_factor
        self.path_output = path_output
        self.optimizer = None

    def optimize(self, masks_fmap):
        """
        Optimization of the currents for each shim group. Wraps :meth:`shimmingtoolbox.shim.sequencer.Sequencer.opt`.

        Args:
            masks_fmap (np.ndarray): 3D fieldmap mask used for the optimizer to shim in the region
                                     of interest (only consider voxels with non-zero values)
        Returns:
                np.ndarray: Coefficients of the coil profiles to shim (len(slices) x n_channels)
        """

        n_shims = len(self.slices)
        coefs = []
        for i in range(n_shims):
            # If there is nothing to shim in this shim group
            if np.all(masks_fmap[..., i] == 0):
                coefs.append(np.zeros(self.optimizer.merged_coils.shape[-1]))

            # Otherwise optimize
            else:
                coefs.append(self.optimizer.optimize(masks_fmap[..., i]))

        return np.array(coefs)


class ShimSequencer(Sequencer):
    """
    ShimSequencer object to perform optimization of shim parameters for static and dynamic shimming. This object can
    also evaluate the shimming performance.

    Attributes:
        nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 3d.
        nii_anat (nib.Nifti1Image): Nibabel object containing anatomical data in 3d.
        nii_mask_anat (nib.Nifti1Image): 3D anat mask used for the optimizer to shim in the region of interest.
                                             (only consider voxels with non-zero values)
        coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                          matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                          Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil profiles
                          are larger than the extent of the fieldmap. This is especially true for dimensions with only
                          1 voxel(e.g. (50x50x1). Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                          :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog.
                      Note: refer to their specific implementation to know limits of the methods
                      in: :mod:`shimmingtoolbox.optimizer`
        opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                            'mae': mean absolute error, 'std': standard deviation.
        nii_fieldmap_orig (nib.Nifti1Image): Nibabel object containing the copy of the original fieldmap data
        optimizer (Optimizer) : Object that contains everything needed for the optimization.
        fmap_is_extended (bool) : Tells whether the fieldmap has been extended by the object.
        masks_fmap (np.ndarray) : Resampled mask on the original fieldmap
    """

    def __init__(self, nii_fieldmap, nii_anat, nii_mask_anat, slices, coils, method='least_squares', opt_criteria='mse',
                 mask_dilation_kernel='sphere', mask_dilation_kernel_size=3, reg_factor=0, path_output=None):
        """
        Initialization for the ShimSequencer class

        Args:
            nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 3d.
            nii_anat (nib.Nifti1Image): Nibabel object containing anatomical data in 3d.
            nii_mask_anat (nib.Nifti1Image): 3D anat mask used for the optimizer to shim in the region of interest.
                                             (only consider voxels with non-zero values)
            slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape of
                            anat is: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
            coils (ListCoil): List of Coils containing the coil profiles. The coil profiles and the fieldmaps must have
                              matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                              Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil
                              profiles are larger than the extent of the fieldmap. This is especially true for
                              dimensions with only 1 voxel(e.g. (50x50x1).
                              Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                              :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
            method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog.
                          Note: refer to their specific implementation to know limits of the methods
                          in: :mod:`shimmingtoolbox.optimizer`
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for
                                        more details.
            mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                              For example, a kernel of size 3 will dilate the mask by 1 pixel.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
                                artefacts.
        """
        super().__init__(slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor, path_output)
        self.nii_fieldmap, self.nii_fieldmap_orig, self.fmap_is_extended = self.get_fieldmap(nii_fieldmap)
        self.nii_anat = self.get_anat(nii_anat)
        self.nii_mask_anat = self.get_mask(nii_mask_anat)
        self.coils = coils
        if opt_criteria not in allowed_opt_criteria:
            raise ValueError("Criteria for optimization not supported")
        self.opt_criteria = opt_criteria
        self.method = method
        self.masks_fmap = None

    def get_fieldmap(self, nii_fieldmap):
        """
        Get the fieldmap and perform error checking.

        Args:
              nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 3d.

        Returns:
            (tuple): tuple containing:

                * nib.Nifti1Image: Nibabel object containing fieldmap data in 3d.
                * nib.Nifti1Image: Nibabel object containing the copy of the initial fieldmap data in 3d.
                * bool: Boolean indicating if the initial fieldmap has been changed.
        """
        nii_fmap_orig = copy.deepcopy(nii_fieldmap)
        if nii_fmap_orig.get_fdata().ndim != 3:
            if nii_fmap_orig.get_fdata().ndim == 2:
                nii_fmap_orig = nib.Nifti1Image(nii_fmap_orig.get_fdata()[..., np.newaxis], nii_fmap_orig.affine,
                                                header=nii_fmap_orig.header)
                nii_fieldmap = extend_fmap_to_kernel_size(nii_fmap_orig, self.mask_dilation_kernel_size,
                                                          self.path_output)
                extending = True
            else:
                raise ValueError("Fieldmap must be 2d or 3d")
        else:
            extending = False
            for i_axis in range(3):
                if nii_fieldmap.get_fdata().shape[i_axis] < self.mask_dilation_kernel_size:
                    extending = True
            if extending:
                nii_fieldmap = extend_fmap_to_kernel_size(nii_fmap_orig, self.mask_dilation_kernel_size,
                                                          self.path_output)

        return nii_fieldmap, nii_fmap_orig, extending

    def get_anat(self, nii_anat):
        """
        Get the target image and perform error checking.

        Args:
            nii_anat (nib.Nifti1Image): Nibabel object containing anatomical data in 3d.

        Returns:
            nib.Nifti1Image: Nibabel object containing anatomical data in 3d.

        """
        anat = nii_anat.get_fdata()
        if anat.ndim == 3:
            pass
        elif anat.ndim == 4:
            logger.info("Target anatomical is 4d, taking the average and converting to 3d")
            anat = np.mean(anat, axis=3)
            nii_anat = nib.Nifti1Image(anat, nii_anat.affine, header=nii_anat.header)
        else:
            raise ValueError("Target anatomical image must be in 3d or 4d")

        return nii_anat

    def get_mask(self, nii_mask_anat):
        """
        Get the mask and perform error checking.

        Args:
            nii_mask_anat (nib.Nifti1Image): 3D anat mask used for the optimizer to shim in the region
                                              of interest.(only consider voxels with non-zero values)

        Returns:
            nib.Nifti1Image: 3D anat mask used for the optimizer to shim in the region of interest.
                              (Only consider voxels with non-zero values)

        """
        anat = self.nii_anat.get_fdata()
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
            nii_mask_anat = nib.Nifti1Image(tmp_3d.astype(int), nii_mask_anat.affine,
                                            header=nii_mask_anat.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_anat, os.path.join(self.path_output, "fig_3d_mask.nii.gz"))
        else:
            raise ValueError("Mask must be in 3d or 4d")

        if not np.all(nii_mask_anat.shape == anat.shape) or not np.all(
                nii_mask_anat.affine == self.nii_anat.affine):
            logger.debug("Resampling mask on the target anat")
            nii_mask_anat_soft = resample_from_to(nii_mask_anat, self.nii_anat, order=1, mode='grid-constant')
            tmp_mask = nii_mask_anat_soft.get_fdata()
            # Change soft mask into binary mask
            tmp_mask = threshold(tmp_mask, thr=0.001)
            nii_mask_anat = nib.Nifti1Image(tmp_mask, nii_mask_anat_soft.affine, header=nii_mask_anat_soft.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_anat, os.path.join(self.path_output, "mask_static_resampled_on_anat.nii.gz"))

        return nii_mask_anat

    def get_resampled_masks(self):
        """
        This function resamples the mask on the fieldmap and on the dilated fieldmap

        Returns:
            (tuple) : tuple containing:
                * nib.Nifti1Image: Mask resampled and dilated on the fieldmap for the optimization
                * nib.Nifti1Image: Mask resampled on the original fieldmap.
        """

        nii_mask_anat = self.nii_mask_anat
        optimizer = self.optimizer
        slices = self.slices
        dilation_kernel = self.mask_dilation_kernel
        dilation_kernel_size = self.mask_dilation_kernel_size
        path_output = self.path_output
        n_shims = len(slices)
        nii_unshimmed = nib.Nifti1Image(optimizer.unshimmed, optimizer.unshimmed_affine)
        if self.fmap_is_extended:
            # Joblib multiprocessing to resampled the mask
            dilated_mask = Parallel(-1, backend='loky')(
                delayed(resample_mask)(nii_mask_anat, nii_unshimmed, slices[i], dilation_kernel,
                                       dilation_kernel_size, path_output)
                for i in range(n_shims))

            nii_unshimmed = self.nii_fieldmap_orig
            mask = Parallel(-1, backend='loky')(
                delayed(resample_mask)(nii_mask_anat, nii_unshimmed, slices[i])
                for i in range(n_shims))

            # We need to transpose the mask to have the good dimensions
            masks_fmap_dilated = np.array([dilated_mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
            masks_fmap = np.array([mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)

        else:
            # Joblib multiprocessing to resampled the mask
            results_mask = Parallel(-1, backend='loky')(
                delayed(resample_mask)(nii_mask_anat, nii_unshimmed, slices[i], dilation_kernel,
                                       dilation_kernel_size, path_output, return_non_dil_mask=True)
                for i in range(n_shims))

            # We need to transpose the mask to have the good dimensions
            masks_fmap_dilated = np.array([results_mask[it][1].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
            masks_fmap = np.array([results_mask[it][0].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)

        return masks_fmap_dilated, masks_fmap

    def shim(self):
        """
        Performs shimming according to slices using one of the supported optimizers and coil profiles.

        Returns:
            np.ndarray: Coefficients of the coil profiles to shim (len(slices) x n_channels)
        """
        # Select and initialize the optimizer
        self.select_optimizer()

        # Get both resampled masks that will be used in the optimization and in the evaluation of the shim
        masks_fmap_dilated, self.masks_fmap = self.get_resampled_masks()

        # Optimize and get the coefficients
        coefs = self.optimize(masks_fmap_dilated)
        return coefs

    def select_optimizer(self):
        """
        Select and initialize the optimizer

        Returns:
            Optimizer: Initialized Optimizer object
        """

        # global supported_optimizers
        if self.method in supported_optimizers:
            if self.method == 'least_squares':
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine, self.opt_criteria,
                                                              reg_factor=self.reg_factor)
            elif self.method == 'quad_prog':
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine, reg_factor=self.reg_factor)
            else:
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine)
        else:
            raise KeyError(f"Method: {self.method} is not part of the supported optimizers")

        self.optimizer = optimizer

    def eval(self, coef):
        """
        Calculate theoretical shimmed map and output figures.

        Args :
            coef (np.ndarray): Coefficients of the coil profiles to shim (len(slices) x n_channels)
        """

        # Save the merged coil profiles if in debug
        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            # Save coils
            nii_merged_coils = nib.Nifti1Image(self.optimizer.merged_coils, self.nii_fieldmap_orig.affine,
                                               header=self.nii_fieldmap_orig.header)
            nib.save(nii_merged_coils, os.path.join(self.path_output, "merged_coils.nii.gz"))

        unshimmed = self.nii_fieldmap_orig.get_fdata()

        # If the fieldmap was changed (i.e. only 1 slice) we want to evaluate the output on the original fieldmap
        if self.fmap_is_extended:
            merged_coils, _ = self.optimizer.merge_coils(unshimmed, self.nii_fieldmap_orig.affine)

        else:
            merged_coils = self.optimizer.merged_coils

        shimmed, corrections, list_shim_slice = self.evaluate_shimming(unshimmed, coef, merged_coils)

        if self.path_output is not None:
            # fmap space
            # Merge the i_shim into one single fieldmap shimmed (correction applied only where it will be applied on
            # the fieldmap)
            shimmed_masked, mask_full_binary = self.calc_shimmed_full_mask(unshimmed, corrections)

            if len(self.slices) == 1:
                # TODO: Output json sidecar
                # TODO: Update the shim settings if Scanner coil?
                # Output the resulting fieldmap since it can be calculated over the entire fieldmap
                nii_shimmed_fmap = nib.Nifti1Image(shimmed[..., 0], self.nii_fieldmap_orig.affine,
                                                   header=self.nii_fieldmap_orig.header)
                fname_shimmed_fmap = os.path.join(self.path_output, 'fieldmap_calculated_shim.nii.gz')
                nib.save(nii_shimmed_fmap, fname_shimmed_fmap)
            else:
                # Output the resulting masked fieldmap since it cannot be calculated over the entire fieldmap
                nii_shimmed_fmap = nib.Nifti1Image(shimmed_masked, self.nii_fieldmap_orig.affine,
                                                   header=self.nii_fieldmap_orig.header)
                fname_shimmed_fmap = os.path.join(self.path_output, 'fieldmap_calculated_shim_masked.nii.gz')
                nib.save(nii_shimmed_fmap, fname_shimmed_fmap)

            # TODO: Add units if possible
            # TODO: Add in anat space?
            # Figure that shows unshimmed vs shimmed for each slice
            self.plot_full_mask(unshimmed, shimmed_masked, mask_full_binary)

            # Figure that shows shim correction for each shim group
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                self.plot_partial_mask(unshimmed, shimmed)

            self.plot_currents(coef)
            self.calc_shimmed_anat_orient(coef, list_shim_slice)
            if logger.level <= getattr(logging, 'DEBUG'):
                # Save to a NIfTI
                fname_correction = os.path.join(self.path_output, 'fig_correction.nii.gz')
                nii_correction_3d = nib.Nifti1Image(shimmed_masked, self.optimizer.unshimmed_affine)
                nib.save(nii_correction_3d, fname_correction)

                # 4th dimension is i_shim
                fname_correction = os.path.join(self.path_output, 'fig_shimmed_4thdim_ishim.nii.gz')
                nii_correction = nib.Nifti1Image(self.masks_fmap * shimmed, self.optimizer.unshimmed_affine)
                nib.save(nii_correction, fname_correction)

    def evaluate_shimming(self, unshimmed, coef, merged_coils):
        """
        Evaluate the shimming and print the efficiency of the corrections.

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            coef (np.ndarray): Coefficients of the coil profiles to shim (len(slices) x n_channels)
            merged_coils (np.ndarray): Coils resampled on the original fieldmap

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Shimmed fieldmap
                * np.ndarray: Corrections to apply to the fieldmap
                * list: List containing the indexes of the shimmed slices
        """
        # Initialize
        list_shim_slice = []
        for i_shim in range(len(self.slices)):
            if np.any(coef[i_shim]):
                list_shim_slice.append(i_shim)
        # Calculate shimmed values
        # This is doing this, but in a faster way by avoiding the for loop :
        # for i_shim in range(len(slices)):
        # corrections[..., i_shim] = merged_coils @ coef[i_shim]
        # shimmed[..., i_shim] = corrections[..., i_shim] + unshimmed
        corrections = np.einsum('ijkl,lm->ijkm', merged_coils, coef.T, optimize='optimizer')
        shimmed = np.add(corrections, np.expand_dims(unshimmed, axis=3))
        self.display_shimmed_results(shimmed, unshimmed, coef)

        return shimmed, corrections, list_shim_slice

    def display_shimmed_results(self, shimmed, unshimmed, coef):
        """
        Print the efficiency of the corrections according to the opt_criteria

        Args:
            shimmed (np.ndarray): Shimmed fieldmap
            unshimmed (np.ndarray): Original fieldmap not shimmed
            coef (np.ndarray): Coefficients of the coil profiles to shim (len(slices) x n_channels)
        """

        for i_shim in range(len(self.slices)):
            mask = np.where(self.masks_fmap[..., i_shim], False, True)
            ma_shimmed = np.ma.array(shimmed[..., i_shim], mask=mask, dtype=np.float32)
            ma_unshimmed = np.ma.array(unshimmed, mask=mask, dtype=np.float32)

            if logger.level <= getattr(logging, 'DEBUG'):
                # Log shimmed results
                mse_shimmed = np.ma.mean(np.square(ma_shimmed))
                mse_unshimmed = np.ma.mean(np.square(ma_unshimmed))
                mae_shimmed = np.ma.mean(np.abs(ma_shimmed))
                mae_unshimmed = np.ma.mean(np.abs(ma_unshimmed))
                std_shimmed = np.ma.std(ma_shimmed)
                std_unshimmed = np.ma.std(ma_unshimmed)

                if mae_unshimmed < mae_shimmed and self.opt_criteria == 'mae':
                    logger.warning("Evaluating the mae, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif std_unshimmed < std_shimmed and self.opt_criteria == 'std':
                    logger.warning("Evaluating the std, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif mse_unshimmed < mse_shimmed:
                    # self.opt_criteria is None or self.opt_criteria == 'mse'
                    logger.warning("Evaluating the mse, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")

                logger.debug(f"Slice(s): {self.slices[i_shim]}\n"
                             f"MAE:\n"
                             f"unshimmed: {mae_unshimmed}, shimmed: {mae_shimmed}\n"
                             f"MSE:\n"
                             f"unshimmed: {mse_unshimmed}, shimmed: {mse_shimmed}\n"
                             f"STD:\n"
                             f"unshimmed: {std_unshimmed}, shimmed: {std_shimmed}\n"
                             f"current: \n{coef[i_shim, :]}")

            else:
                # Log shimmied results only if they are worse than no shimming
                if self.opt_criteria == 'mae':
                    mae_shimmed = np.ma.mean(np.abs(ma_shimmed))
                    mae_unshimmed = np.ma.mean(np.abs(ma_unshimmed))
                    if mae_unshimmed < mae_shimmed:
                        logger.warning("Evaluating the mae, verify the shim parameters."
                                       " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif self.opt_criteria == 'std':
                    std_shimmed = np.ma.std(ma_shimmed)
                    std_unshimmed = np.ma.std(ma_unshimmed)
                    if std_unshimmed < std_shimmed:
                        logger.warning("Evaluating the std, verify the shim parameters."
                                       " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                else:
                    # self.opt_criteria is None or self.opt_criteria == 'mse' or ...
                    mse_shimmed = np.ma.mean(np.square(ma_shimmed))
                    mse_unshimmed = np.ma.mean(np.square(ma_unshimmed))
                    if mse_unshimmed < mse_shimmed:
                        logger.warning("Evaluating the mse. Verify the shim parameters."
                                       " Some give worse results than no shim.\n " f"i_shim: {i_shim}")

    def calc_shimmed_full_mask(self, unshimmed, correction):
        """
        Calculate the shimmed full mask

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            correction (np.ndarray): Corrections to apply to the fieldmap
        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Masked shimmed fieldmap
                * np.ndarray: Binary mask in the fieldmap space
        """
        mask_full_binary = np.clip(np.ceil(resample_from_to(self.nii_mask_anat,
                                                            self.nii_fieldmap_orig,
                                                            order=0,
                                                            mode='grid-constant',
                                                            cval=0).get_fdata()), 0, 1)
        # Find the correction
        # This is the same as this but in a faster way:
        # for i_shim in range(len(slices)):
        #   full_correction += correction[..., i_shim] * masks_fmap[..., i_shim]
        full_correction = np.einsum('ijkl,ijkl->ijk', self.masks_fmap, correction, optimize='optimizer')
        # Calculate the weighted whole mask
        mask_weight = np.sum(self.masks_fmap, axis=3)
        # Divide by the weighted mask. This is done so that the edges of the soft mask can be shimmed appropriately
        full_correction_scaled = np.divide(full_correction, mask_weight, where=mask_full_binary.astype(bool))

        # Apply the correction to the unshimmed image
        shimmed_masked = (full_correction_scaled + unshimmed) * mask_full_binary

        return shimmed_masked, mask_full_binary

    def plot_full_mask(self, unshimmed, shimmed_masked, mask):
        """
        Plot and save the static full mask

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            shimmed_masked(np.ndarray): Masked shimmed fieldmap
            mask (np.ndarray): Binary mask in the fieldmap space
        """
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
        fname_figure = os.path.join(self.path_output, 'fig_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure, bbox_inches='tight')

    def plot_partial_mask(self, unshimmed, shimmed):
        """
        This figure shows a single fieldmap slice for all shim groups. The shimmed and unshimmed fieldmaps are in
        the background and the correction is overlaid in color.

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            shimmed (np.ndarray): Shimmed fieldmap
        """
        a_slice = 0
        unshimmed_repeated = unshimmed[..., np.newaxis] * np.ones(self.masks_fmap.shape[-1])
        mt_unshimmed = montage(unshimmed_repeated[:, :, a_slice, :])
        mt_shimmed = montage(shimmed[:, :, a_slice, :])

        unshimmed_masked = unshimmed_repeated * np.greater(self.masks_fmap, 0)
        mt_unshimmed_masked = montage(unshimmed_masked[:, :, a_slice, :])
        mt_shimmed_masked = montage(shimmed[:, :, a_slice, :] * np.ceil(self.masks_fmap[:, :, a_slice, :]))

        min_masked_value = np.min([mt_unshimmed_masked, mt_shimmed_masked])
        max_masked_value = np.max([mt_unshimmed_masked, mt_shimmed_masked])

        min_fmap_value = np.min([mt_unshimmed, mt_shimmed])
        max_fmap_value = np.max([mt_unshimmed, mt_shimmed])

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
        fname_figure = os.path.join(self.path_output, 'fig_shimmed_vs_unshimmed_shim_groups.png')
        fig.savefig(fname_figure, bbox_inches='tight')

    def plot_currents(self, static):
        """
        Plot evolution of currents through shim groups

        Args:
            static (np.ndarray): Array with the static coefficients
        """
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        n_channels = static.shape[1]
        for i_channel in range(n_channels):
            ax.plot(static[:, i_channel], label=f"Static channel{i_channel} currents through shim groups")
        ax.set_xlabel('Shim group')
        ax.set_ylabel('Coefficients (Physical CS [RAS])')
        ax.legend()
        ax.set_title("Currents through shims")
        fname_figure = os.path.join(self.path_output, 'fig_currents.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def calc_shimmed_anat_orient(self, coefs, list_shim_slice):
        """
        Calculate and save the shimmed anat orient

        Args:
            coefs (np.ndarray): Coefficients of the coil profiles to shim (len(slices) x n_channels)
            list_shim_slice (list): list of the index where there was a correction
        """
        # TODO: resample shimmed fieldmap using order 1 to the target coord system
        nii_coils = nib.Nifti1Image(self.optimizer.merged_coils, self.nii_fieldmap_orig.affine,
                                    header=self.nii_fieldmap_orig.header)
        coils_anat = resample_from_to(nii_coils,
                                      self.nii_mask_anat,
                                      order=1,
                                      mode='grid-constant',
                                      cval=0).get_fdata()
        fieldmap_anat = resample_from_to(self.nii_fieldmap_orig,
                                         self.nii_mask_anat,
                                         order=1,
                                         mode='grid-constant',
                                         cval=0).get_fdata()

        shimmed_anat_orient = copy.deepcopy(fieldmap_anat)
        for i_shim in list_shim_slice:
            corr = np.sum(coefs[i_shim] * coils_anat[:, :, self.slices[i_shim], :], axis=3, keepdims=False)
            shimmed_anat_orient[..., self.slices[i_shim]] += corr

        fname_shimmed_anat_orient = os.path.join(self.path_output, 'fig_shimmed_anat_orient.nii.gz')
        nii_shimmed_anat_orient = nib.Nifti1Image(shimmed_anat_orient * self.nii_mask_anat.get_fdata(),
                                                  self.nii_mask_anat.affine,
                                                  header=self.nii_mask_anat.header)
        nib.save(nii_shimmed_anat_orient, fname_shimmed_anat_orient)


class RealTimeSequencer(Sequencer):
    """
    Sequencer object that stores different nibabel object, and parameters. It's also doing real time optimization
    of the currents, and the evaluation of the shimming

    Attributes:
            nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension
                                            is the timeseries. Also contains an affine transformation.
            json_fmap (dict): Dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                              timestamps).
            nii_anat (nib.Nifti1Image): Nibabel object containing anatomical data in 3d.
            nii_static_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                              component.
            nii_riro_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                              component.
            slices (list): 1D array containing tuples of dim3 slices to shim according to the anat where the shape of
                            anat: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
            coils (ListCoil): List of `Coils` containing the coil profiles. The coil profiles and the fieldmaps must
                              have matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                              Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil
                              profiles are larger than the extent of the fieldmap. This is especially true for
                              dimensions with only 1 voxel(e.g. (50x50x1x10).
                              Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                              :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
            method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog.
                          Note: refer to their specific implementation to know limits of the methods
                          in: :mod:`shimmingtoolbox.optimizer`
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for
                                        more details.
            mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                             For example, a kernel of size 3 will dilate the mask by 1 pixel.
            path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
                               artefacts.
            optimizer (object) : Object that contains everything needed for the optimization created from
                                `shimmingtoolbox.optimizer` init method
            optimizer_riro (object) : Object that contains everything needed for the riro optimization created from
                                `shimmingtoolbox.optimizer` init method
            bounds (list) : List of the bounds for the currents for the real time optimization
            acq_pressures (np.ndarray) : 1D array that contains the acquisitions pressures
    """

    def __init__(self, nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices, pmu: PmuResp, coils,
                 method='least_squares', opt_criteria='mse', mask_dilation_kernel='sphere', mask_dilation_kernel_size=3,
                 reg_factor=0, path_output=None):
        """
        Initialization of the RealTimeSequencer class

        Args:
            nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension
                                            is the timeseries. Also contains an affine transformation.
            json_fmap (dict): Dict of the json sidecar corresponding to the fieldmap data (Used to find the acquisition
                              timestamps).
            nii_anat (nib.Nifti1Image): Nibabel object containing anatomical data in 3d.
            nii_static_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the static
                                               component.
            nii_riro_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                             component.
            slices (list): 1D array containing tuples of dim3 slices to shim according to the anat where the shape of
                            anat: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
            coils (ListCoil): List of `Coils` containing the coil profiles. The coil profiles and the fieldmaps must
                              have matching units (if fmap is in Hz, the coil profiles must be in hz/unit_shim).
                              Refer to :class:`shimmingtoolbox.coils.coil.Coil`. Make sure the extent of the coil
                              profiles are larger than the extent of the fieldmap. This is especially true for
                              dimensions with only 1 voxel(e.g. (50x50x1x10).
                              Refer to :func:`shimmingtoolbox.shim.sequencer.extend_slice`/
                              :func:`shimmingtoolbox.shim.sequencer.update_affine_for_ap_slices`
            method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog.
                          Note: refer to their specific implementation to know limits of the methods
                          in: :mod:`shimmingtoolbox.optimizer`
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.dilate_binary_mask` for
                                        more details.
            mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                             For example, a kernel of size 3 will dilate the mask by 1 pixel.

        """
        super().__init__(slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor, path_output)
        self.json_fmap = json_fmap
        self.pmu = pmu
        self.coils = coils
        self.method = method
        self.bounds = None

        if opt_criteria not in allowed_opt_criteria:
            raise ValueError("Criteria for optimization not supported")

        self.opt_criteria = opt_criteria
        self.nii_fieldmap, self.nii_fieldmap_orig = self.get_fieldmap(nii_fieldmap)

        # Check if anat has the good dimensions
        if nii_anat.get_fdata().ndim != 3:
            raise ValueError("Anatomical image must be in 3d")

        self.nii_anat = nii_anat
        self.nii_static_mask, self.nii_riro_mask = self.get_mask(nii_static_mask, nii_riro_mask)
        self.acq_pressures = self.get_acq_pressures()
        self.optimizer_riro = None

    def get_fieldmap(self, nii_fieldmap):
        """
        Get the fieldmap for the RealTimeSequencer class

        Args:
           nii_fieldmap (nib.Nifti1Image): Nibabel object containing fieldmap data in 4d where the 4th dimension
                                           is the timeseries.

        Returns:
            nib.Nifti1Image: Nibabel object containing fieldmap data in 4d where the 4th dimension
                             is the timeseries.

        """
        # Make sure the fieldmap has the appropriate dimensions
        if nii_fieldmap.get_fdata().ndim != 4:
            raise ValueError("Fieldmap must be 4d (dim1, dim2, dim3, t)")

        nii_fmap_orig = copy.deepcopy(nii_fieldmap)
        # Extend the fieldmap if there are axes that have less voxels than the kernel size. This is done since we are
        # fitting a fieldmap to coil profiles and having a small number of voxels can lead to errors in fitting (2
        # voxels in one dimension can differentiate order 1 at most), the parameter allows to have at least the size
        # of the kernel for each dimension This is usually useful in the through plane direction where we could have
        # less slices. To mitigate this, we create a 3d volume by replicating the slices on the edges.
        for i_axis in range(3):
            if nii_fmap_orig.shape[i_axis] < self.mask_dilation_kernel_size:
                nii_fieldmap = extend_fmap_to_kernel_size(nii_fmap_orig, self.mask_dilation_kernel_size,
                                                          self.path_output)
                break

        return nii_fieldmap, nii_fmap_orig

    def get_mask(self, nii_static_mask, nii_riro_mask):
        """
            Get both masks for the RealTimeSequencer Class

            Args:
                nii_static_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region
                                                   for the static component.
                nii_riro_mask (nib.Nifti1Image): 3D anat mask used for the optimizer to shim the region for the riro
                                                 component.

            Returns:
                (tuple) : tuple containing:
                    * nib.Nifti1Image: 3D anat mask used for the optimizer to shim the region for the static component.
                    * nib.Nifti1Image: 3D anat mask used for the optimizer to shim the region for the riro component.

            """
        # Note: We technically don't need the anat if we use the nii_mask. However, this is a nice safety check to
        # make sure the mask is indeed in the dimension of the anat and not the fieldmap.

        anat = self.nii_anat.get_fdata()
        # Make sure masks have the appropriate dimensions
        if nii_static_mask.get_fdata().ndim != 3:
            raise ValueError("static_mask image must be in 3d")
        if nii_riro_mask.get_fdata().ndim != 3:
            raise ValueError("riro_mask image must be in 3d")

        # Resample the input masks on the target anatomical image if they are different
        if not np.all(nii_static_mask.shape == anat.shape) or not np.all(
                nii_static_mask.affine == self.nii_anat.affine):
            logger.debug("Resampling static mask on the target anat")
            nii_static_mask_soft = resample_from_to(nii_static_mask, self.nii_anat, order=1, mode='grid-constant')
            tmp_mask = nii_static_mask_soft.get_fdata()
            # Change soft mask into binary mask
            tmp_mask = threshold(tmp_mask, thr=0.001)
            nii_static_mask = nib.Nifti1Image(tmp_mask, nii_static_mask_soft.affine,
                                              header=nii_static_mask_soft.header)

            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_static_mask, os.path.join(self.path_output, "mask_static_resampled_on_anat.nii.gz"))

        if not np.all(nii_riro_mask.shape == anat.shape) or not np.all(
                nii_riro_mask.affine == self.nii_anat.affine):
            logger.debug("Resampling riro mask on the target anat")
            nii_riro_mask_soft = resample_from_to(nii_riro_mask, self.nii_anat, order=1, mode='grid-constant')
            tmp_mask = nii_riro_mask_soft.get_fdata()
            # Change soft mask into binary mask
            tmp_mask = threshold(tmp_mask, thr=0.001)
            nii_riro_mask = nib.Nifti1Image(tmp_mask, nii_riro_mask_soft.affine, header=nii_riro_mask_soft.header)

            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_riro_mask, os.path.join(self.path_output, "mask_riro_resampled_on_anat.nii.gz"))

        return nii_static_mask, nii_riro_mask

    def get_acq_pressures(self):
        """
        Get the acquisition pressures at the times when the fieldmap was acquired.

        Returns:
            np.ndarray: 1D array that contains the acquisitions pressures
        """
        # Fetch PMU timing
        acq_timestamps = get_acquisition_times(self.nii_fieldmap, self.json_fmap)
        # TODO: deal with saturation
        # fit PMU and fieldmap values
        acq_pressures = self.pmu.interp_resp_trace(acq_timestamps)

        return acq_pressures

    def get_real_time_parameters(self):
        """
        Get real time parameters used for the shimming

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 3D array containing the static data for the optimization
                * np.ndarray: 3D array containing the real time data for the optimization
                * float: Mean pressure of the respiratory trace.
                * float: Root mean squared of the pressure trace. This is provided to compare results between scans,
                         multiply the riro coefficients by rms of the pressure to do so.

        """
        fieldmap = self.nii_fieldmap.get_fdata()
        anat = self.nii_anat.get_fdata()
        # regularization --> static, riro
        # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
        mean_p = np.mean(self.acq_pressures)
        pressure_rms = np.sqrt(np.mean((self.acq_pressures - mean_p) ** 2))
        x = self.acq_pressures.reshape(-1, 1) - mean_p

        # Safety check for linear regression if the pressure and fieldmap fit well
        # Mask the voxels not being shimmed for riro
        nii_3dfmap = nib.Nifti1Image(self.nii_fieldmap.get_fdata()[..., 0], self.nii_fieldmap.affine,
                                     header=self.nii_fieldmap.header)
        fmap_mask_riro = resample_mask(self.nii_riro_mask, nii_3dfmap, tuple(range(anat.shape[2])),
                                       dilation_kernel=self.mask_dilation_kernel,
                                       dilation_size=self.mask_dilation_kernel_size).get_fdata()
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
        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            # Save static
            nii_static = nib.Nifti1Image(static, self.nii_fieldmap.affine, header=self.nii_fieldmap.header)
            nib.save(nii_static, os.path.join(self.path_output, 'fig_static_fmap_component.nii.gz'))

            # Save riro
            nii_riro = nib.Nifti1Image(riro, self.nii_fieldmap.affine, header=self.nii_fieldmap.header)
            nib.save(nii_riro, os.path.join(self.path_output, 'fig_riro_fmap_component.nii.gz'))

        return static, riro, mean_p, pressure_rms

    def shim(self):
        """
        Performs realtime shimming using one of the supported optimizers and an external respiratory trace.

        Returns:
            (tuple): tuple containing:
                * np.ndarray: Static coefficients of the coil profiles to shim (len(slices) x channels) e.g. [Hz]
                * np.ndarray: Riro coefficients of the coil profiles to shim (len(slices) x channels)
                              e.g. [Hz/unit_pressure]
                * float: Mean pressure of the respiratory trace.
                * float: Root mean squared of the pressure.
                         This is provided to compare results between scans, multiply the riro coefficients
                         by rms of the pressure to do so.
        """

        affine_fieldmap = self.nii_fieldmap.affine
        static, riro, mean_p, pressure_rms = self.get_real_time_parameters()

        # Create both optimizer object
        self.select_optimizer(static, affine_fieldmap)
        if self.method == 'least_squares':
            self.method = 'least_squares_rt'
        if self.method == 'quad_prog':
            self.method = 'quad_prog_rt'
        self.select_optimizer(riro, affine_fieldmap, self.pmu)

        # Create both resampled masks used for the optimization
        static_mask_resampled, riro_mask_resampled = self.get_riro_and_static_resampled_masks()

        # Static shim
        logger.info("Static optimization")
        coef_static = self.optimize(static_mask_resampled)

        # RIRO optimization
        # Use the currents to define a list of new coil bounds for the riro optimization
        self.bounds = new_bounds_from_currents(coef_static, self.optimizer.merged_bounds)

        logger.info("Realtime optimization")
        coef_riro = self.optimize_riro(riro_mask_resampled)

        # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
        # between scans. This allows to compare results between scans.
        # coef_riro_rms = coef_riro * pressure_rms
        # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

        return coef_static, coef_riro, mean_p, pressure_rms

    def select_optimizer(self, unshimmed, affine, pmu: PmuResp = None):
        """
        Select and initialize the optimizer

        Args:
            unshimmed (np.ndarray): 3D B0 map
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            pmu (PmuResp): PmuResp object containing the respiratory trace information. Required for method
                           'least_squares_rt'.

        """

        # global supported_optimizers
        if self.method in supported_optimizers:
            if self.method == 'least_squares':
                self.optimizer = supported_optimizers[self.method](self.coils, unshimmed, affine, self.opt_criteria,
                                                                   reg_factor=self.reg_factor)
            elif self.method == 'quad_prog':
                self.optimizer = supported_optimizers[self.method](self.coils, unshimmed, affine,
                                                                   reg_factor=self.reg_factor)

            elif self.method == 'least_squares_rt':
                # Make sure pmu is defined
                if pmu is None:
                    raise ValueError(f"pmu parameter is required if using the optimization method: {self.method}")

                # Add pmu to the realtime optimizer(s)
                self.optimizer_riro = supported_optimizers[self.method](self.coils, unshimmed, affine,
                                                                        self.opt_criteria, pmu,
                                                                        reg_factor=self.reg_factor)
            elif self.method == 'quad_prog_rt':
                # Make sure pmu is defined
                if pmu is None:
                    raise ValueError(f"pmu parameter is required if using the optimization method: {self.method}")

                # Add pmu to the realtime optimizer(s)
                self.optimizer_riro = supported_optimizers[self.method](self.coils, unshimmed, affine, pmu,
                                                                        reg_factor=self.reg_factor)

            else:
                if pmu is None:
                    self.optimizer = supported_optimizers[self.method](self.coils, unshimmed, affine)
                else:
                    self.optimizer_riro = supported_optimizers[self.method](self.coils, unshimmed, affine)

        else:
            raise KeyError(f"Method: {self.method} is not part of the supported optimizers")

    def get_riro_and_static_resampled_masks(self):
        """
        This function resample the static and the riro masks on the differents elements needed
         for the optimization and the evaluation

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Static mask resampled and dilated on the fieldmap for the optimization
                * np.ndarray: Riro mask resampled and dilated on the original fieldmap for the optimization.
        """

        n_shims = len(self.slices)
        dilation_kernel = self.mask_dilation_kernel
        dilation_kernel_size = self.mask_dilation_kernel_size
        slices = self.slices
        path_output = self.path_output
        nii_riro_mask = self.nii_riro_mask
        nii_static_mask = self.nii_static_mask

        static_fieldmap = nib.Nifti1Image(self.optimizer.unshimmed, self.optimizer.unshimmed_affine)
        riro_fieldmap = nib.Nifti1Image(self.optimizer_riro.unshimmed, self.optimizer_riro.unshimmed_affine)

        static_mask = Parallel(-1, backend='loky')(
            delayed(resample_mask)(nii_static_mask, static_fieldmap, slices[i],
                                   dilation_kernel,
                                   dilation_kernel_size, path_output)
            for i in range(n_shims))
        riro_mask = Parallel(-1, backend='loky')(
            delayed(resample_mask)(nii_riro_mask, riro_fieldmap, slices[i],
                                   dilation_kernel,
                                   dilation_kernel_size, path_output)
            for i in range(n_shims))

        static_mask_resampled = np.array([static_mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
        riro_mask_resampled = np.array([riro_mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)

        return static_mask_resampled, riro_mask_resampled

    def optimize_riro(self, mask_anat):
        """
        Args:
            mask_anat (np.ndarray): anat mask on which the optimization will be made
        Returns:
            Riro coefficients of the coil profiles to shim (len(slices) x channels) [Hz/unit_pressure]
        """
        # It's faster to use local arguments for the optimization
        n_shims = len(self.slices)
        optimizer = self.optimizer_riro
        shimwise_bounds = self.bounds
        coefs_riro = []
        for i in range(n_shims):
            # Change bounds
            if shimwise_bounds is not None:
                optimizer.set_merged_bounds(shimwise_bounds[i])
            # Return 0s if there is no optimization to perform
            if np.all(mask_anat[..., i] == 0):
                coefs_riro.append(np.zeros(optimizer.merged_coils.shape[-1]))
            # Optimize
            else:
                coefs_riro.append(optimizer.optimize(mask_anat[..., i]))

        return np.array(coefs_riro)

    def eval(self, coef_static, coef_riro, mean_p, pressure_rms):
        """
        Evaluate the real time shimming by plotting and saving results

        Args:
            coef_static (np.ndarray): coefficients got during the static optimization
            coef_riro (np.ndarray): coefficients got during the real time optimization
            mean_p (float): mean of the acquisitions pressures
            pressure_rms (float): rms of the acquisitions pressures
        """

        logger.debug("Calculating the sum of the shimmed vs unshimmed in the static ROI.")
        # Calculate theoretical shimmed map
        # shim
        unshimmed = self.nii_fieldmap.get_fdata()
        nii_target = nib.Nifti1Image(self.nii_fieldmap.get_fdata()[..., 0], self.nii_fieldmap.affine,
                                     header=self.nii_fieldmap.header)
        shape = unshimmed.shape + (len(self.slices),)
        shimmed_static_riro = np.zeros(shape)
        shimmed_static = np.zeros(shape)
        shimmed_riro = np.zeros(shape)
        masked_shim_static_riro = np.zeros(shape)
        masked_shim_static = np.zeros(shape)
        masked_shim_riro = np.zeros(shape)
        masked_unshimmed = np.zeros(shape)
        mask_fmap_cs = np.zeros(unshimmed[..., 0].shape + (len(self.slices),))
        shim_trace_static_riro = []
        shim_trace_static = []
        shim_trace_riro = []
        unshimmed_trace = []
        mask_full_binary = np.clip(np.ceil(resample_from_to(self.nii_static_mask,
                                                            nii_target,
                                                            order=0,
                                                            mode='grid-constant',
                                                            cval=0).get_fdata()), 0, 1)
        for i_shim in range(len(self.slices)):
            # Calculate static correction
            correction_static = self.optimizer_riro.merged_coils @ coef_static[i_shim]

            # Calculate the riro coil profiles
            riro_profile = self.optimizer_riro.merged_coils @ coef_riro[i_shim]

            mask_fmap_cs[..., i_shim] = np.ceil(resample_mask(self.nii_static_mask, nii_target,
                                                              self.slices[i_shim]).get_fdata())
            for i_t in range(self.nii_fieldmap.shape[3]):
                # Apply the static and riro correction
                correction_riro = riro_profile * (self.acq_pressures[i_t] - mean_p)
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
                # TODO: Calculate the sum of mask_fmap_cs[..., i_shim] and divide by that (If the roi is bigger due to
                #  interpolation, it should not count more). Possibly use soft mask?
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
                             f"Riro currents:\n{coef_riro[i_shim] * (self.acq_pressures[i_t] - mean_p)}\n")

                # Create a 1D list of the sum of the shimmed and unshimmed maps
                shim_trace_static.append(sum_shimmed_static)
                shim_trace_static_riro.append(sum_shimmed_static_riro)
                shim_trace_riro.append(sum_shimmed_riro)
                unshimmed_trace.append(sum_unshimmed)

        # reshape to slice x timepoint
        nt = unshimmed.shape[3]
        n_shim = len(self.slices)
        shim_trace_static = np.array(shim_trace_static).reshape(n_shim, nt)
        shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_shim, nt)
        shim_trace_riro = np.array(shim_trace_riro).reshape(n_shim, nt)
        unshimmed_trace = np.array(unshimmed_trace).reshape(n_shim, nt)

        if self.path_output is not None:
            # Plot before vs after shimming averaged on time
            shimmed_mask_avg = np.zeros(mask_full_binary.shape)
            np.divide(np.sum(np.mean(masked_shim_static_riro, axis=3), axis=3), np.sum(mask_fmap_cs, axis=3), where=mask_full_binary.astype(bool), out=shimmed_mask_avg)
            self.plot_full_mask(np.mean(unshimmed, axis=3), shimmed_mask_avg, mask_full_binary)

            # Plot STD over time before and after shimming
            self.plot_full_time_std(unshimmed, masked_shim_static_riro, mask_fmap_cs, mask_full_binary)

        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            # plot results
            i_slice = 0
            i_shim = 0
            i_t = 0
            while np.all(masked_unshimmed[..., i_slice, i_t, i_shim] == np.zeros(masked_unshimmed.shape[:2])):
                i_shim += 1
                if i_shim >= n_shim - 1:
                    break
            self.plot_static_riro(masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed,
                                  shimmed_static,
                                  shimmed_static_riro, i_slice=i_slice, i_shim=i_shim, i_t=i_t)
            self.plot_currents(coef_static, riro=coef_riro * pressure_rms)
            self.plot_shimmed_trace(unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro)
            self.plot_pressure_points(self.acq_pressures, (self.pmu.min, self.pmu.max))
            self.print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, mask_fmap_cs)
            # Save shimmed result
            nii_shimmed_static_riro = nib.Nifti1Image(shimmed_static_riro, self.nii_fieldmap.affine,
                                                      header=self.nii_fieldmap.header)
            nib.save(nii_shimmed_static_riro, os.path.join(self.path_output,
                                                           'shimmed_static_riro_4thdim_it_5thdim_ishim.nii.gz'))

            # Save coils
            nii_merged_coils = nib.Nifti1Image(self.optimizer_riro.merged_coils, self.nii_fieldmap.affine,
                                               header=self.nii_fieldmap.header)
            nib.save(nii_merged_coils, os.path.join(self.path_output, "merged_coils.nii.gz"))

    def plot_currents(self, static, riro=None):
        """
        Plot evolution of currents through shim groups

        Args:
            static (np.ndarray): Array with the static currents
            riro (np.ndarray): Array with the riro currents
        """
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
        fname_figure = os.path.join(self.path_output, 'fig_currents.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def plot_static_riro(self, masked_unshimmed, masked_shim_static, masked_shim_static_riro, unshimmed,
                         shimmed_static, shimmed_static_riro, i_t=0, i_slice=0, i_shim=0):
        """
        Plot Static and RIRO maps for a particular fieldmap slice, anat shim and timepoint

        Args:
            masked_unshimmed (np.ndarray):  Fieldmap masked before the shimming
            masked_shim_static (np.ndarray): Fieldmap masked after static shimming
            masked_shim_static_riro (np.ndarray): Fieldmap masked after the static and riro shimming
            unshimmed (np.ndarray): Fieldmap not shimmed
            shimmed_static (np.ndarray): Data of the nii_fieldmap after the static shimming
            shimmed_static_riro (np.ndarray): Data of the nii_fieldmap after static and riro shimming
            i_shim: (int): index of the anat shim, where we want to plot the static and riro maps
            i_slice: (int): index of the slice, where we want to plot the static and riro maps
            i_t: (int): Index of the time, where we want to plot the static and riro maps
        """

        min_value = min(masked_shim_static_riro[..., i_slice, i_t, i_shim].min(),
                        masked_shim_static[..., i_slice, i_t, i_shim].min(),
                        masked_unshimmed[..., i_slice, i_t, i_shim].min())
        max_value = max(masked_shim_static_riro[..., i_slice, i_t, i_shim].max(),
                        masked_shim_static[..., i_slice, i_t, i_shim].max(),
                        masked_unshimmed[..., i_slice, i_t, i_shim].max())

        index_slice_to_show = self.slices[i_shim][i_slice]

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
        fname_figure = os.path.join(self.path_output, 'fig_realtime_masked_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def plot_pressure_points(self, acq_pressures, ylim):
        """
        Plot respiratory trace pressure points

        Args:
            acq_pressures (np.ndarray): acquisitions pressures
            ylim (float): Limit of the y-axis

        """
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.plot(acq_pressures, label='pressures')
        ax.legend()
        ax.set_ylim(ylim)
        ax.set_title("Pressures vs time points")
        fname_figure = os.path.join(self.path_output, 'fig_trace_pressures.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def plot_shimmed_trace(self, unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro):
        """
        Plot shimmed and unshimmed sum over the roi for each shim

        Args:
            unshimmed_trace (np.ndarray): array with the trace of the nii_fieldmap data
            shim_trace_static (np.ndarray): array with the trace of the nii_fieldmap data after the static shimming
            shim_trace_riro (np.ndarray): array with the trace of the nii_fieldmap data after the riro shimming
            shim_trace_static_riro (np.ndarray): array with the trace of the nii_fieldmap data after both shimming
        """

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
        fname_figure = os.path.join(self.path_output, 'fig_trace_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def print_rt_metrics(self, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, mask):
        """
        Print to the console metrics about the realtime and static shim. These metrics isolate temporal and static
        components

        Temporal: Compute the STD across time pixelwise, and then compute the mean across pixels.
        Static: Compute the MEAN across time pixelwise, and then compute the STD across pixels.

        Args:
            unshimmed (np.ndarray): Fieldmap not shimmed
            shimmed_static (np.ndarray): Data of the nii_fieldmap after the static shimming
            shimmed_static_riro (np.ndarray): Data of the nii_fieldmap after static and riro shimming
            shimmed_riro (np.ndarray): Data of the nii_fieldmap after the riro shimming
            mask (np.ndarray): Mask where the shimming was done
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

    def plot_full_mask(self, unshimmed, shimmed_masked, mask):
        """
        Plot and save the static full mask

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            shimmed_masked(np.ndarray): Masked shimmed fieldmap
            mask (np.ndarray): Binary mask in the fieldmap space
        """
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
        fname_figure = os.path.join(self.path_output, 'fig_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure, bbox_inches='tight')

    def plot_full_time_std(self, unshimmed, masked_shim_static_riro, mask_fmap_cs, mask):
        """
        Plot and save the std heatmap over time

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed shaped (x, y, z, time)
            masked_shim_static_riro(np.ndarray): Masked shimmed fieldmap shaped (x, y, z, time, slices)
            mask_fmap_cs (np.ndarray): Field map mask indicating where delta B0 is not 0 in each slice -- shaped (x, y, z, slices)
            mask (np.ndarray): Binary mask in the fieldmap space shaped (x, y, z)
        """
        # Transform shimmed field map to shape (x, y, z, time)
        sum_mask_fmap_cs =  np.sum(mask_fmap_cs, axis=3)
        mask_extended = np.repeat(mask[..., np.newaxis], masked_shim_static_riro.shape[-2], axis=-1)

        # Transpose is used to cater to numpy division order
        # (3, 2, 4) / (3, 2) Does not work
        # (4, 2, 3) / (2, 3) Does work
        #* Using out parameter in np.divide() prevents inconsistent results
        shimmed_masked = np.zeros(mask_extended.shape)
        np.divide(np.sum(masked_shim_static_riro, axis=-1).T, sum_mask_fmap_cs.T, where=mask.T.astype(bool), out=shimmed_masked.T)

        std_shimmed_masked = np.std(shimmed_masked, axis=-1, dtype=np.float64)
        std_unshimmed = np.std(unshimmed, axis=-1, dtype=np.float64)

        ## Plot
        mt_unshimmed = montage(np.mean(unshimmed, axis=-1))
        mt_unshimmed_masked = montage(std_unshimmed * mask)
        mt_shimmed_masked = montage(std_shimmed_masked)

        metric_unshimmed_mean = calculate_metric_within_mask(std_unshimmed, mask, metric='mean')
        metric_shimmed_mean = calculate_metric_within_mask(std_shimmed_masked, mask, metric='mean')

        # Remove the outliners to calculate the colorbar limits
        # Necessary because some STD are much higher and are not visible on the heatmap, they are still considered in the metric
        shim_limit = np.percentile(mt_shimmed_masked[mt_shimmed_masked !=0], 90)
        unshim_limit = np.percentile(mt_unshimmed_masked[mt_unshimmed_masked !=0], 90)

        min_value = min(mt_unshimmed_masked.min(), mt_shimmed_masked.min())
        max_value = max(mt_unshimmed_masked[mt_unshimmed_masked < unshim_limit].max(), mt_shimmed_masked[mt_shimmed_masked < shim_limit].max())

        fig = Figure(figsize=(9, 6))
        fig.suptitle(f"Fieldmaps\nFieldmap Coordinate System\n\u0394B\u2080 STD over time ")

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(mt_unshimmed, cmap='gray')
        mt_unshimmed_masked[mt_unshimmed_masked == 0] = np.nan
        im = ax.imshow(mt_unshimmed_masked, vmin=min_value, vmax=max_value, cmap='viridis')
        ax.set_title(f"Before shimming\nmean: {metric_unshimmed_mean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(mt_unshimmed, cmap='gray')
        mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
        im = ax.imshow(mt_shimmed_masked, vmin=min_value, vmax=max_value, cmap='viridis')
        ax.set_title(f"After shimming\nmean: {metric_shimmed_mean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        # Lower suptitle
        fig.subplots_adjust(top=0.85)

        # Save
        fname_figure = os.path.join(self.path_output, 'fig_shimmed_vs_unshimmed_real-time_variation.png')
        fig.savefig(fname_figure, bbox_inches='tight')


def new_bounds_from_currents(currents, old_bounds):
    """
    Uses the currents to determine the appropriate bounds for the next optimization. It assumes that
    "old_coef + next_bound < old_bound".

    Args:
        currents (np.ndarray): 2D array (n_shims x n_channels).
        old_bounds (list): 2D list (n_channels, 2) containing (min, max) containing the merged bounds of the previous
                           optimization.
    Returns:
        list: 2d list (n_shim_groups x n_channels) of bounds (min, max) corresponding to each shim group and channel.
    """

    new_bounds = []
    for i_shim in range(currents.shape[0]):
        shim_bound = []
        for i_channel in range(len(old_bounds)):
            if old_bounds[i_channel] == [None, None]:
                a_bound = old_bounds[i_channel]
            elif old_bounds[i_channel][0] is None:
                a_bound = [None, old_bounds[i_channel][1] - currents[i_shim, i_channel]]
            elif old_bounds[i_channel][1] is None:
                a_bound = [old_bounds[i_channel][0] - currents[i_shim, i_channel], None]
            else:
                a_bound = [old_bounds[i_channel][0] - currents[i_shim, i_channel],
                           old_bounds[i_channel][1] - currents[i_shim, i_channel]]
            shim_bound.append(a_bound)
        new_bounds.append(shim_bound)

    return new_bounds


def parse_slices(fname_nifti):
    """
    Parse the BIDS sidecar associated with the input nifti file.

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
        raise RuntimeError("No tag SliceTiming to automatically parse slice data, see --slices option")

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
    """
    Find indexes of the 4th dimension of the input volume that has the highest signal intensity for each slice.
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


def extend_fmap_to_kernel_size(nii_fmap_orig, dilation_kernel_size, path_output=None):
    """
    Load the fmap and expand its dimensions to the kernel size

    Args:
        nii_fmap_orig (nib.Nifti1Image): 3d (dim1, dim2, dim3) or 4d (dim1, dim2, dim3, t) nii to be extended
        dilation_kernel_size: Size of the kernel
        path_output (str): Path to save the debug output
    Returns:
        nib.Nifti1Image: Nibabel object of the loaded and extended fieldmap
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
    """
    Adds n_slices on each side of the selected axis. It uses the nearest slice and copies it to fill the values.
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


def update_affine_for_ap_slices(affine, n_slices=1, axis=2):
    """
    Updates the input affine to reflect an insertion of n_slices on each side of the selected axis

    Args:
        affine (np.ndarray): 4x4 qform affine matrix representing the coordinates
        n_slices (int): Number of pixels to add on each side of the selected axis
        axis (int): Axis along which to insert the slice(s)
    Returns:
        np.ndarray: 4x4 updated affine matrix
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
