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
from scipy.signal import find_peaks, savgol_filter

from shimmingtoolbox.masking.mask_utils import modify_binary_mask
from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer, PmuLsqOptimizer, allowed_opt_criteria
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.optimizer.quadprog_optimizer import QuadProgOpt, PmuQuadProgOpt
from shimmingtoolbox.coils.coil import Coil, ScannerCoil, SCANNER_CONSTRAINTS, SCANNER_CONSTRAINTS_DAC
from shimmingtoolbox.coils.spher_harm_basis import channels_per_order
from shimmingtoolbox.optimizer.bfgs_optimizer import BFGSOpt, PmuBFGSOpt
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.masking.mask_utils import resample_mask
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.utils import create_output_dir, montage
from shimmingtoolbox.shim.shim_utils import calculate_metric_within_mask

ListCoil = List[Coil]

logger = logging.getLogger(__name__)

supported_optimizers = {
    'least_squares_rt': PmuLsqOptimizer,
    'least_squares': LsqOptimizer,
    'quad_prog': QuadProgOpt,
    'quad_prog_rt': PmuQuadProgOpt,
    'bfgs': BFGSOpt,
    'bfgs_rt': PmuBFGSOpt,
    'pseudo_inverse': Optimizer,
}

GAMMA = 42.576E6  # in Hz/Tesla


class Sequencer(object):
    """
    General class for the sequencer

    Attributes:
        slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape
                       of anat is: (dim1, dim2, dim3). Refer to :func:`shimmingtoolbox.shim.sequencer.define_slices`.
        mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                    'cube'. See :func:`shimmingtoolbox.masking.mask_utils.modify_binary_mask` for more
                                    details.
        mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                         For example, a kernel of size 3 will dilate the mask by 1 pixel.
        reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                            penalize higher current values while a lower factor will lower the effect of the
                            regularization. A negative value will favour high currents (not preferred). Only relevant
                            for 'least_squares' opt_method.
        path_output (str): Path to the directory to output figures. Set logging level to debug to output debug
        index_shimmed: Indexes of ``slices`` that have been shimmed
        index_not_shimmed: Indexes of ``slices`` that have not been shimmed
    """

    def __init__(self, slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor,
                 w_signal_loss=0, w_signal_loss_xy=0, epi_te=0, path_output=None):
        """
        Constructor of the sequencer class

        Args:
            slices (list): 1D array containing tuples of dim3 slices to shim according to the anat, where the shape
                           of anat is: (dim1, dim2, dim3). Refer to
                           :func:`shimmingtoolbox.shim.sequencer.define_slices`.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.modify_binary_mask` for
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
        self.w_signal_loss = w_signal_loss
        self.w_signal_loss_xy = w_signal_loss_xy
        self.epi_te = epi_te
        self.optimizer = None
        self.index_shimmed = []
        self.index_not_shimmed = []

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
                self.index_not_shimmed.append(i)

            # Otherwise optimize
            else:
                coefs.append(self.optimizer.optimize(masks_fmap[..., i]))
                self.index_shimmed.append(i)

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
        method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog', 'bfgs'.
                      Note: refer to their specific implementation to know limits of the methods
                      in: :mod:`shimmingtoolbox.optimizer`
        opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                            'mae': mean absolute error, 'std': standard deviation, 'ps_huber': pseudo huber cost function.
        nii_fieldmap_orig (nib.Nifti1Image): Nibabel object containing the copy of the original fieldmap data
        optimizer (Optimizer) : Object that contains everything needed for the optimization.
        fmap_is_extended (bool) : Tells whether the fieldmap has been extended by the object.
        masks_fmap (np.ndarray) : Resampled mask on the original fieldmap
    """

    def __init__(self, nii_fieldmap, json_fieldmap, nii_anat, json_anat, nii_mask_anat, slices, coils,
                 method='least_squares', opt_criteria='mse',
                 mask_dilation_kernel='sphere', mask_dilation_kernel_size=3, reg_factor=0, w_signal_loss=None,
                 w_signal_loss_xy=None, epi_te=None, path_output=None):
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
            method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog', 'bfgs'.
                          Note: refer to their specific implementation to know limits of the methods
                          in: :mod:`shimmingtoolbox.optimizer`
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation, 'rmse': root mean squared error,
                                'ps_huber': pseudo huber cost function.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.modify_binary_mask` for
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
        super().__init__(slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor, path_output=path_output)
        self.nii_fieldmap, self.nii_fieldmap_orig, self.fmap_is_extended = self.get_fieldmap(nii_fieldmap)
        self.json_fieldmap = json_fieldmap
        self.nii_anat = self.get_anat(nii_anat)
        self.json_anat = json_anat
        self.nii_mask_anat = self.load_masks(nii_mask_anat)
        self.coils = coils
        if opt_criteria not in allowed_opt_criteria:
            raise ValueError("Criteria for optimization not supported")
        self.opt_criteria = opt_criteria
        self.method = method
        self.masks_fmap = None
        self.w_signal_loss = w_signal_loss
        self.w_signal_loss_xy = w_signal_loss_xy
        self.epi_te = epi_te

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

    def load_masks(self, nii_mask_anat):
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
            nii_mask_anat = nib.Nifti1Image(tmp_3d.astype(int), nii_mask_anat.affine, header=nii_mask_anat.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_anat, os.path.join(self.path_output, "fig_3d_mask.nii.gz"))
        else:
            raise ValueError("Mask must be in 3d or 4d")

        # Check if the mask needs to be resampled
        if not np.all(nii_mask_anat.shape == anat.shape) or not np.all(nii_mask_anat.affine == self.nii_anat.affine):
            # Resample the mask on the target anatomical image
            logger.debug("Resampling mask on the target anat")
            nii_mask_anat = resample_from_to(nii_mask_anat, self.nii_anat, order=1, mode='grid-constant')
            # Save the resampled mask
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
            masks_fmap_dilated = np.array([results_mask[it][1].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3,
                                                                                                                0)
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
            if self.method in ['least_squares', 'bfgs']:
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine, self.opt_criteria,
                                                              reg_factor=self.reg_factor,
                                                              w_signal_loss=self.w_signal_loss,
                                                              w_signal_loss_xy=self.w_signal_loss_xy,
                                                              epi_te=self.epi_te)
            elif self.method == 'quad_prog':
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine, reg_factor=self.reg_factor)
            else:
                optimizer = supported_optimizers[self.method](self.coils, self.nii_fieldmap.get_fdata(),
                                                              self.nii_fieldmap.affine)
        else:
            raise KeyError(f"Method: {self.method} is not part of the supported optimizers")

        self.optimizer = optimizer

    def eval(self, coefs):
        """
        Calculate theoretical shimmed map and output figures.

        Args :
            coefs (np.ndarray): Coefficients of the coil profiles to shim (len(slices) x n_channels)
        """

        # Save the merged coil profiles if in debug

        unshimmed = self.nii_fieldmap_orig.get_fdata()

        # If the fieldmap was changed (i.e. only 1 slice) we want to evaluate the output on the original fieldmap
        if self.fmap_is_extended:
            merged_coils, _ = self.optimizer.merge_coils(unshimmed, self.nii_fieldmap_orig.affine)
        else:
            merged_coils = self.optimizer.merged_coils

        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            if self.fmap_is_extended:
                # Save coils with extended slices
                nii_merged_coils = nib.Nifti1Image(self.optimizer.merged_coils, self.nii_fieldmap.affine,
                                                   header=self.nii_fieldmap.header)
                nib.save(nii_merged_coils, os.path.join(self.path_output, "merged_coils_opt.nii.gz"))

            # Save coil with original dimensions
            nii_merged_coils = nib.Nifti1Image(merged_coils, self.nii_fieldmap_orig.affine,
                                               header=self.nii_fieldmap_orig.header)
            nib.save(nii_merged_coils, os.path.join(self.path_output, "merged_coils.nii.gz"))

        shimmed, corrections, list_shim_slice = self.evaluate_shimming(unshimmed, coefs, merged_coils)
        shimmed_masked, mask_full = self.calc_shimmed_full_mask(unshimmed, corrections)
        if self.path_output is not None:
            # fmap space
            if len(self.slices) == 1:
                # Output the resulting fieldmap since it can be calculated over the entire fieldmap
                nii_shimmed_fmap = nib.Nifti1Image(shimmed[..., 0], self.nii_fieldmap_orig.affine,
                                                   header=self.nii_fieldmap_orig.header)
                fname_shimmed_fmap = os.path.join(self.path_output, 'fieldmap_calculated_shim.nii.gz')
                nib.save(nii_shimmed_fmap, fname_shimmed_fmap)

            else:
                # Output the resulting masked fieldmap since it cannot be calculated over the entire fieldmap
                nii_shimmed_fmap = nib.Nifti1Image(shimmed_masked, self.nii_fieldmap_orig.affine,
                                                   header=self.nii_fieldmap_orig.header)
                fname_shimmed_fmap = os.path.join(self.path_output, 'fieldmap_calculated_shim.nii.gz')
                nib.save(nii_shimmed_fmap, fname_shimmed_fmap)

            # Output JSON file
            self.save_calc_fmap_json(coefs)

            # TODO: Add units if possible
            # TODO: Add in anat space?
            if 'signal_recovery' in self.opt_criteria:

                full_Gz = np.zeros(corrections.shape)
                full_Gx = np.zeros(corrections.shape)
                full_Gy = np.zeros(corrections.shape)
                shimmed_temp = corrections + unshimmed[..., np.newaxis]

                # Can't calculate signal recovery in the through slice direction if there is only one slice
                if corrections.shape[2] != 1:
                    full_Gz = np.gradient(shimmed_temp, axis=2)
                    full_Gz, _ = self.calc_shimmed_gradient_full_mask(full_Gz)
                    # Plot gradient results
                    self._plot_static_signal_recovery_mask(unshimmed, full_Gz, mask_full)

                full_Gx = np.gradient(shimmed_temp, axis=0)
                full_Gy = np.gradient(shimmed_temp, axis=1)
                full_Gx, _ = self.calc_shimmed_gradient_full_mask(full_Gx)
                full_Gy, _ = self.calc_shimmed_gradient_full_mask(full_Gy)

                if logger.level <= getattr(logging, 'DEBUG'):
                    # x, y, z are in the patient's coordinate system
                    if corrections.shape[2] != 1:
                        self._plot_G_mask(np.gradient(unshimmed, axis=2), full_Gz, mask_full, name='Gz')
                    self._plot_G_mask(np.gradient(unshimmed, axis=0), full_Gx, mask_full, name='Gx')
                    self._plot_G_mask(np.gradient(unshimmed, axis=1), full_Gy, mask_full, name='Gy')

                    # Resample the shimmed fieldmap and the corrections (useful for the evaluation of the shim)
                    shimmed_temp_nii = nib.Nifti1Image(shimmed_temp, affine=self.nii_fieldmap_orig.affine,
                                                        header=self.nii_fieldmap_orig.header)
                    corrections_nii = nib.Nifti1Image(corrections, affine=self.nii_fieldmap_orig.affine,
                                                    header=self.nii_fieldmap_orig.header)
                    shimmed_temp_resample_nii = resample_from_to(shimmed_temp_nii, self.nii_anat, order=1, mode='grid-constant')
                    corrections_resample_nii = resample_from_to(corrections_nii, self.nii_anat, order=1, mode='grid-constant')
                    nib.save(shimmed_temp_resample_nii, os.path.join(self.path_output, 'fieldmap_calculated_shim_resampled.nii.gz'))
                    nib.save(corrections_resample_nii, os.path.join(self.path_output, 'corrections_resampled.nii.gz'))
                    # TODO: Output JSON file, since it is resampled, the JSON from the fmap might not be appropriate

            # Figure that shows unshimmed vs shimmed for each slice
            plot_full_mask(unshimmed, shimmed_masked, mask_full, self.path_output)

            # Figure that shows shim correction for each shim group
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                # The 0th slice is selected here, but can be changed for debugging purposes
                self.plot_partial_mask(unshimmed, shimmed, slice=0)

            self.plot_currents(coefs)

            self.calc_shimmed_anat_orient(coefs, list_shim_slice)
            if logger.level <= getattr(logging, 'DEBUG'):
                # Save to a NIfTI
                fname_correction = os.path.join(self.path_output, 'fig_correction_i_shim.nii.gz')
                nii_correction_3d = nib.Nifti1Image(corrections, self.optimizer.unshimmed_affine)
                nib.save(nii_correction_3d, fname_correction)

                fname_correction = os.path.join(self.path_output, 'fig_correction.nii.gz')
                nii_correction_3d = nib.Nifti1Image(np.sum(corrections, axis=3), self.optimizer.unshimmed_affine)
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

            mask = self.masks_fmap[..., i_shim]
            if np.sum(mask) == 0:
                continue
            i_shimmed = shimmed[..., i_shim]

            if logger.level <= getattr(logging, 'DEBUG'):
                # Log shimmed results
                mse_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'mse')
                mse_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'mse')
                mae_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'mae')
                mae_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'mae')
                std_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'std')
                std_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'std')

                if mae_unshimmed < mae_shimmed and self.opt_criteria == 'mae':
                    logger.warning("Evaluating the mae, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif std_unshimmed < std_shimmed and self.opt_criteria == 'std':
                    logger.warning("Evaluating the std, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif mse_unshimmed < mse_shimmed:
                    logger.warning("Evaluating the mse, verify the shim parameters."
                                   " Some give worse results than no shim.\n " f"i_shim: {i_shim}")

                logger.debug(f"Slice(s): {self.slices[i_shim]}\n"
                             f"MAE:\n"
                             f"unshimmed: {mae_unshimmed}, shimmed: {mae_shimmed}\n"
                             f"MSE:\n"
                             f"unshimmed: {mse_unshimmed}, shimmed: {mse_shimmed}\n"
                             f"RMSE:\n"
                             f"unshimmed: {np.sqrt(mse_unshimmed)}, shimmed: {np.sqrt(mse_shimmed)}\n"
                             f"STD:\n"
                             f"unshimmed: {std_unshimmed}, shimmed: {std_shimmed}\n"
                             f"current: \n{coef[i_shim, :]}")

            else:
                # Log shimmied results only if they are worse than no shimming
                if self.opt_criteria == 'mae':
                    mae_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'mae')
                    mae_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'mae')
                    if mae_unshimmed < mae_shimmed:
                        logger.warning("Evaluating the mae, verify the shim parameters."
                                       " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                elif self.opt_criteria == 'std':
                    std_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'std')
                    std_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'std')
                    if std_unshimmed < std_shimmed:
                        logger.warning("Evaluating the std, verify the shim parameters."
                                       " Some give worse results than no shim.\n " f"i_shim: {i_shim}")
                else :
                    mse_shimmed = calculate_metric_within_mask(i_shimmed, mask, 'mse')
                    mse_unshimmed = calculate_metric_within_mask(unshimmed, mask, 'mse')
                    if mse_unshimmed < mse_shimmed:
                        logger.warning("Evaluating the mse, verify the shim parameters."
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
                * np.ndarray: Mask in the fieldmap space
        """
        mask_full = np.clip(resample_from_to(self.nii_mask_anat,
                                        self.nii_fieldmap_orig,
                                        order=0,
                                        mode='grid-constant',
                                        cval=0).get_fdata(), 0, 1)
        mask_full_binary = (mask_full != 0).astype(int)

        full_correction = np.einsum('ijkl,ijkl->ijk', self.masks_fmap, correction, optimize='optimizer')

        # Calculate the weighted whole mask
        mask_weight = np.sum(self.masks_fmap, axis=3)

        # Divide by the weighted mask. This is done so that the edges of the soft mask can be shimmed appropriately
        full_correction_scaled = np.divide(full_correction, mask_weight, where=mask_full_binary.astype(bool))

        # Apply the correction to the unshimmed image
        shimmed_masked = (full_correction_scaled + unshimmed) * mask_full_binary

        return shimmed_masked, mask_full

    def calc_shimmed_gradient_full_mask(self, gradient):
        """
        Calculate the shimmed gradient full mask

        Args:
            gradient (np.ndarray): Gradient of each shimmed fieldmap slice
        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Masked shimmed fieldmap
                * np.ndarray: Mask in the fieldmap space
        """
        mask_full = np.clip(resample_from_to(self.nii_mask_anat,
                                                    self.nii_fieldmap_orig,
                                                    order=0,
                                                    mode='grid-constant',
                                                    cval=0).get_fdata(), 0, 1)
        mask_full_binary = (mask_full != 0).astype(int)

        full_correction = np.einsum('ijkl,ijkl->ijk', self.masks_fmap, gradient, optimize='optimizer')

        # Calculate the weighted whole mask
        mask_weight = np.sum(self.masks_fmap, axis=3)

        # Divide by the weighted mask. This is done so that the edges of the soft mask can be shimmed appropriately
        full_correction_scaled = np.divide(full_correction, mask_weight, where=mask_full_binary.astype(bool))

        # Apply the correction to the unshimmed image
        shimmed_masked = full_correction_scaled * mask_full_binary

        return shimmed_masked, mask_full

    def plot_partial_mask(self, unshimmed, shimmed, slice):
        """
        This figure shows a single fieldmap slice for all shim groups. The shimmed and unshimmed fieldmaps are in
        the background and the correction is overlaid in color.

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed
            shimmed (np.ndarray): Shimmed fieldmap
            slice (int): Slice to plot
        """
        # Binarize the mask
        bin_mask = (self.masks_fmap != 0).astype(int)

        unshimmed_repeated = unshimmed[..., np.newaxis] * np.ones(self.masks_fmap.shape[-1])
        nan_unshimmed_masked = np.ma.array(unshimmed_repeated, mask=(bin_mask == 0), fill_value=np.nan)
        nan_shimmed_masked = np.ma.array(shimmed, mask=(bin_mask == 0), fill_value=np.nan)

        mt_unshimmed = montage(unshimmed_repeated[:, :, slice, :])
        mt_shimmed = montage(shimmed[:, :, slice, :])
        mt_unshimmed_masked = montage(nan_unshimmed_masked[:, :, slice, :].filled())
        mt_shimmed_masked = montage(nan_shimmed_masked[:, :, slice, :].filled() * np.ceil(self.masks_fmap[:, :, slice, :]))

        min_masked_value = np.nanmin([mt_unshimmed_masked, mt_shimmed_masked])
        max_masked_value = np.nanmax([mt_unshimmed_masked, mt_shimmed_masked])
        min_fmap_value = np.nanmin([mt_unshimmed, mt_shimmed])
        max_fmap_value = np.nanmax([mt_unshimmed, mt_shimmed])

        fig = Figure(figsize=(15, 9))
        fig.suptitle(f"Slice {slice} fieldmap for all shim groups\nFieldmap Coordinate System")

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(mt_unshimmed, vmin=min_fmap_value, vmax=max_fmap_value, cmap='gray')
        im = ax.imshow(mt_unshimmed_masked, vmin=min_masked_value, vmax=max_masked_value, cmap='viridis')
        ax.set_title("Before shimming")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(mt_shimmed, vmin=min_fmap_value, vmax=max_fmap_value, cmap='gray')
        im = ax.imshow(mt_shimmed_masked, vmin=min_masked_value, vmax=max_masked_value, cmap='viridis')
        ax.set_title("After shimming")
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

    def save_calc_fmap_json(self, coefs):
        json_shimmed = copy.deepcopy(self.json_fieldmap)
        if len(self.slices) == 1:
            # i keeps track of the index of the concatenated shim coefficients
            i = 0
            for coil in self.coils:
                # j keeps track of the index of the order
                j = 0
                if isinstance(coil, ScannerCoil):
                    # If its volume shim (len(slices == 1)) and a scanner coil
                    # Dump the shim coefficients as ShimSettingsCurrent + calculated shimmed coefs
                    if 0 in coil.orders:
                        json_shimmed['ImagingFrequency'] = int(coil.coefs_used['0'] + coefs[0, i]) / 1e6
                        j += 1
                    shim_settings_output = []
                    for order in (1, 2, 3):
                        if order in coil.orders:
                            manufacturer = self.json_fieldmap.get('Manufacturer')
                            n_channels = channels_per_order(order, manufacturer)
                            for i_channel in range(n_channels):
                                if coil.coefs_used[str(order)] is not None and coil.coefs_used[str(order)][i_channel] is not None:
                                    shim_settings_tmp = (coil.coefs_used[str(order)][i_channel] +
                                                         coefs[0, i + j + i_channel])
                                    manufacturers_model_name = self.json_fieldmap.get('ManufacturersModelName')
                                    if manufacturers_model_name is not None:
                                        manufacturers_model_name = manufacturers_model_name.replace(' ', '_')
                                    if manufacturer in SCANNER_CONSTRAINTS_DAC.keys() \
                                            and manufacturers_model_name in SCANNER_CONSTRAINTS_DAC[manufacturer].keys() \
                                            and str(order) in SCANNER_CONSTRAINTS_DAC[manufacturer][
                                        manufacturers_model_name].keys() \
                                            and manufacturer in SCANNER_CONSTRAINTS.keys() \
                                            and manufacturers_model_name in SCANNER_CONSTRAINTS[manufacturer].keys() \
                                            and str(order) in SCANNER_CONSTRAINTS[manufacturer][
                                        manufacturers_model_name].keys():
                                        scanner_constraints_dac = SCANNER_CONSTRAINTS_DAC[manufacturer][
                                            manufacturers_model_name][str(order)][i_channel]
                                        scanner_constraints_ui = SCANNER_CONSTRAINTS[manufacturer][
                                            manufacturers_model_name][str(order)][i_channel]

                                        # This is where Siemens shim units are converted back to DAC units
                                        shim_settings_tmp = (np.array(shim_settings_tmp) * 2 * np.array(scanner_constraints_dac) /
                                                             (scanner_constraints_ui[1] - scanner_constraints_ui[0]))
                                        tolerance = 0.001 * scanner_constraints_dac
                                        if (shim_settings_tmp > (scanner_constraints_dac + tolerance)) or \
                                                (shim_settings_tmp < (-scanner_constraints_dac - tolerance)):
                                            logger.warning(
                                                f"Future shim settings: order {order}, channel {i_channel} exceeds "
                                                f"known system limits.")

                                    elif manufacturer == 'Siemens':
                                        logger.warning("Scanner constraints not implemented. "
                                                       "Output fieldmap Shim Settings will not be populated.")
                                        shim_settings_tmp = None

                                    shim_settings_output.append(shim_settings_tmp)
                                else:
                                    shim_settings_output.append(None)
                            j += n_channels

                    formatted_shim_settings = []
                    for st in shim_settings_output:
                        if st is not None:
                            formatted_shim_settings.append(float(f"{st:.6g}"))
                        else:
                            formatted_shim_settings.append(None)
                    json_shimmed['ShimSetting'] = formatted_shim_settings

                i += coil.dim[3]

        with open(os.path.join(self.path_output, "fieldmap_calculated_shim.json"), "w") as outfile:
            json.dump(json_shimmed, outfile, indent=4)

    def _plot_static_signal_recovery_mask(self, unshimmed, shimmed_Gz, mask):
        # Plot signal loss maps
        def calculate_signal_loss(gradient):
            slice_thickness = self.json_anat['SliceThickness']
            B0_map_thickness = self.nii_fieldmap.header['pixdim'][3]
            phi = 2 * math.pi * gradient / B0_map_thickness * self.epi_te * slice_thickness
            # The /pi is because the sinc function in numpy is sinc(x) = sin(pi*x)/(pi*x)
            signal_map = abs(np.sinc(phi / (2 * math.pi)))
            signal_loss_map = 1 - signal_map
            return signal_loss_map

        unshimmed_signal_loss = calculate_signal_loss(np.gradient(unshimmed, axis=2))
        shimmed_signal_loss = calculate_signal_loss(shimmed_Gz)

        # Convert soft mask into binary mask
        bin_mask = (mask != 0).astype(int)

        bin_mask_erode = modify_binary_mask(bin_mask, shape='sphere', size=3, operation='erode')
        mask_erode = mask * bin_mask_erode

        # choose selected slices to plot
        nonzero_indices = np.nonzero(np.sum(bin_mask_erode, axis=(0, 1)))[0]
        mt_unshimmed_masked = montage(unshimmed_signal_loss[:, :, nonzero_indices] * bin_mask_erode[:, :, nonzero_indices])
        mt_shimmed_masked = montage(shimmed_signal_loss[:, :, nonzero_indices] * bin_mask_erode[:, :, nonzero_indices])

        nib.save(nib.Nifti1Image(unshimmed_signal_loss, affine=self.nii_fieldmap.affine, header=self.nii_fieldmap.header),
                 os.path.join(self.path_output, 'signal_loss_unshimmed.nii.gz'))
        nib.save(nib.Nifti1Image(shimmed_signal_loss, affine=self.nii_fieldmap.affine, header=self.nii_fieldmap.header),
                 os.path.join(self.path_output, 'signal_loss_shimmed.nii.gz'))
        nib.save(nib.Nifti1Image(mask_erode, affine=self.nii_fieldmap.affine, header=self.nii_fieldmap.header),
                 os.path.join(self.path_output, 'mask_erode.nii.gz'))

        temp_unshimmed_signal_loss = unshimmed_signal_loss.copy()
        temp_unshimmed_signal_loss[unshimmed_signal_loss < 0.1] = np.nan
        temp_shimmed_signal_loss = shimmed_signal_loss.copy()
        temp_shimmed_signal_loss[unshimmed_signal_loss < 0.1] = np.nan

        metric_unshimmed_std = calculate_metric_within_mask(temp_unshimmed_signal_loss, mask_erode, metric='std')
        metric_shimmed_std = calculate_metric_within_mask(temp_shimmed_signal_loss, mask_erode, metric='std')
        metric_unshimmed_mean = calculate_metric_within_mask(temp_unshimmed_signal_loss, mask_erode, metric='mean')
        metric_shimmed_mean = calculate_metric_within_mask(temp_shimmed_signal_loss, mask_erode, metric='mean')
        metric_unshimmed_absmean = calculate_metric_within_mask(np.abs(temp_unshimmed_signal_loss), mask_erode, metric='mean')
        metric_shimmed_absmean = calculate_metric_within_mask(np.abs(temp_shimmed_signal_loss), mask_erode, metric='mean')

        fig = Figure(figsize=(15, 9))
        fig.suptitle("Signal Percentage Loss Map\nFieldmap Coordinate System")

        ax = fig.add_subplot(1, 2, 1)
        mt_unshimmed_masked[mt_shimmed_masked == 0] = np.nan

        im = ax.imshow(mt_unshimmed_masked, vmin=0, vmax=1, cmap='hot')
        ax.set_title(f"Before shimming signal loss \nSTD: {metric_unshimmed_std:.3}, mean: {metric_unshimmed_mean:.3}, "
                     f"abs mean: {metric_unshimmed_absmean:.3}")

        # Change title font size
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)  # signal loss map should be from [0, 1]

        ax = fig.add_subplot(1, 2, 2)
        mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
        im = ax.imshow(mt_shimmed_masked, vmin=0, vmax=1, cmap='hot')
        ax.set_title(f"After shimming signal loss \nSTD: {metric_shimmed_std:.3}, mean: {metric_shimmed_mean:.3}, "
                     f"abs mean: {metric_shimmed_absmean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)  # signal loss map should be from [0, 1]

        # Save
        fname_figure = os.path.join(self.path_output, 'fig_signal_loss_metric_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure, bbox_inches='tight')

    def _plot_G_mask(self, unshimmed_G, shimmed_G, mask, name='G'):
        # Plot Gradient maps

        # Convert soft mask into binary mask
        bin_mask = (mask != 0).astype(int)

        bin_mask_erode = modify_binary_mask(bin_mask, shape='sphere', size=3, operation='erode')
        mask_erode = mask * bin_mask_erode

        # choose selected slices to plot
        nonzero_indices = np.nonzero(np.sum(bin_mask_erode, axis=(0, 1)))[0]
        mt_unshimmed_masked = montage(unshimmed_G[:, :, nonzero_indices] * bin_mask_erode[:, :, nonzero_indices])
        mt_shimmed_masked = montage(shimmed_G[:, :, nonzero_indices] * bin_mask_erode[:, :, nonzero_indices])

        metric_unshimmed_std = calculate_metric_within_mask(unshimmed_G, mask_erode, metric='std')
        metric_shimmed_std = calculate_metric_within_mask(shimmed_G, mask_erode, metric='std')
        metric_unshimmed_mean = calculate_metric_within_mask(unshimmed_G, mask_erode, metric='mean')
        metric_shimmed_mean = calculate_metric_within_mask(shimmed_G, mask_erode, metric='mean')
        metric_unshimmed_absmean = calculate_metric_within_mask(np.abs(unshimmed_G), mask_erode, metric='mean')
        metric_shimmed_absmean = calculate_metric_within_mask(np.abs(shimmed_G), mask_erode, metric='mean')

        fig = Figure(figsize=(15, 9))  # make the figure larger and higher resolution
        fig.suptitle(f"{name}\nFieldmap Coordinate System")

        ax = fig.add_subplot(1, 2, 1)
        mt_unshimmed_masked[mt_shimmed_masked == 0] = np.nan

        im = ax.imshow(mt_unshimmed_masked, vmin=-30, vmax=30, cmap='jet')
        ax.set_title(f"Before shimming {name} \nSTD: {metric_unshimmed_std:.3}, mean: {metric_unshimmed_mean:.3}, "
                     f"abs mean: {metric_unshimmed_absmean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 2, 2)
        mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
        im = ax.imshow(mt_shimmed_masked, vmin=-30, vmax=30, cmap='jet')
        ax.set_title(f"After shimming {name} \nSTD: {metric_shimmed_std:.3}, mean: {metric_shimmed_mean:.3}, "
                     f"abs mean: {metric_shimmed_absmean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        # Save
        fname_figure = os.path.join(self.path_output, f'fig_{name}_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure, bbox_inches='tight')


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
                                'mae': mean absolute error, 'std': standard deviation, 'rmse': root mean squared error.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.modify_binary_mask` for
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
            acq_timestamps (np.ndarray) : 1D array that contains the acquisitions timestamps
            extended_fmap (bool): True if the fieldmap was extended to be able to shim only 1 slice
    """

    def __init__(self, nii_fieldmap, json_fmap, nii_anat, nii_static_mask, nii_riro_mask, slices, pmu: PmuResp,
                 coils_static, coils_riro, method='least_squares', opt_criteria='mse', mask_dilation_kernel='sphere',
                 mask_dilation_kernel_size=3, reg_factor=0, path_output=None, is_pmu_time_offset_auto=False):
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
            method (str): Supported optimizer: 'least_squares', 'pseudo_inverse', 'quad_prog', 'bfgs'.
                          Note: refer to their specific implementation to know limits of the methods
                          in: :mod:`shimmingtoolbox.optimizer`
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
                                Only relevant for 'least_squares' opt_method.
            mask_dilation_kernel (str): Kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                                        'cube'. See :func:`shimmingtoolbox.masking.mask_utils.modify_binary_mask` for
                                        more details.
            mask_dilation_kernel_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd.
                                             For example, a kernel of size 3 will dilate the mask by 1 pixel.
            is_pmu_time_offset_auto (bool): If True, the PMU time offset will be automatically calculated.

        """
        super().__init__(slices, mask_dilation_kernel, mask_dilation_kernel_size, reg_factor, path_output=path_output)
        self.json_fmap = json_fmap
        self.pmu = pmu
        self.coils_static = coils_static
        self.coils_riro = coils_riro
        self.method = method
        self.bounds = None

        if opt_criteria not in allowed_opt_criteria:
            raise ValueError("Criteria for optimization not supported")

        self.opt_criteria = opt_criteria
        self.fmap_orig_location = None
        self.nii_fieldmap, self.nii_fieldmap_orig, self.extended_fmap = self.get_fieldmap(nii_fieldmap)

        # Check if anat has the good dimensions
        if nii_anat.get_fdata().ndim != 3:
            raise ValueError("Anatomical image must be in 3d")

        self.nii_anat = nii_anat
        self.nii_static_mask, self.nii_riro_mask = self.load_masks(nii_static_mask, nii_riro_mask)

        # Resample the masks to the fmap coordinate system for each shim group
        (_,
         _,
         self.mask_static_fmcs_per_shim_dil,
         self.mask_riro_fmcs_per_shim_dil) = self.resample_masks_to_target_per_shim(self.nii_fieldmap)
        (self.mask_static_orig_fmcs_per_shim,
         self.mask_riro_orig_fmcs_per_shim,
         _,
         _) = self.resample_masks_to_target_per_shim(self.nii_fieldmap_orig)

        # Resample the whole masks to the fmap coordinate system
        (self.mask_static_fmcs,
         self.mask_riro_fmcs,
         self.mask_static_fmcs_dil,
         self.mask_riro_fmcs_dil) = self.resample_mask_to_target(self.nii_fieldmap)
        (self.mask_static_orig_fmcs,
         self.mask_riro_orig_fmcs,
         self.mask_static_orig_fmcs_dil,
         self.mask_riro_orig_fmcs_dil) = self.resample_mask_to_target(self.nii_fieldmap_orig)

        self.acq_timestamps = None
        self.acq_timestamps_orig = None
        self.acq_pressures_orig = None
        self.acq_pressures = None
        if is_pmu_time_offset_auto:
            time_offset = self.calculate_best_pmu_time_offset()
            self.pmu.adjust_start_time(time_offset)
        self.get_acq_pressures()
        self.optimizer_riro = None

    def calculate_best_pmu_time_offset(self):
        logger.info(f"Calculating best time offset")
        # Probably sweep at all times but centered on the frequency added
        n_slices = self.nii_fieldmap_orig.shape[2]

        previous_time_offset = self.pmu.time_offset
        self.pmu.adjust_start_time(0)
        mean_respiratory_cycle_time = self.pmu.get_mean_trigger_span() / 2
        acq_times = get_acquisition_times(self.nii_fieldmap_orig, self.json_fmap, when='slice-middle')
        n_samples = 1000
        start_time_mdh, stop_time_mdh = self.pmu.get_start_and_stop_times()
        min_bound_offset = max(-mean_respiratory_cycle_time / 2, start_time_mdh - acq_times.min())
        max_bound_offset = min(mean_respiratory_cycle_time / 2, stop_time_mdh - acq_times.max())
        time_offsets = np.linspace(min_bound_offset, max_bound_offset, n_samples)

        mask_fmap = np.zeros_like(self.mask_static_orig_fmcs)
        mask_fmap[self.mask_riro_orig_fmcs != 0] = self.mask_static_orig_fmcs[self.mask_riro_orig_fmcs != 0]
        mask_4d = np.repeat(np.expand_dims(mask_fmap, axis=-1), self.nii_fieldmap_orig.shape[-1], axis=-1)
        fmap_ma = np.ma.array(self.nii_fieldmap_orig.get_fdata(), mask=mask_4d == False)

        # Find best time offset
        best_r2_total = 0
        best_time_offset = 0
        r2_total_list = []
        for time_offset in time_offsets:
            self.pmu.adjust_start_time(round(time_offset))
            r2_total = 0
            for i_slice in range(n_slices):
                if i_slice in []:
                    continue
                pressures = self.pmu.interp_resp_trace(acq_times) - self.pmu.mean(acq_times.min(), acq_times.max())
                y = fmap_ma.mean(axis=(0, 1))[i_slice].filled()
                y = (y - y.mean())
                reg = LinearRegression().fit(pressures[:, i_slice].reshape(-1, 1), y)
                # Adjusted r2 score
                r2 = reg.score(pressures[:, i_slice].reshape(-1, 1), y)
                # r2_corr = (1 - (1 - r2) * (len(y) - 1) / (len(y) - pressures[:, i_slice].reshape(-1, 1).shape[1] - 1))
                r2_total += r2
            r2_total /= n_slices
            r2_total_list.append(r2_total)
            if best_r2_total < r2_total:
                best_r2_total = r2_total
                best_time_offset = time_offset

        logger.info(f"Best time offset: {round(best_time_offset)}ms")
        logger.info(f"Average r2 score: {best_r2_total} at this time offset")

        if self.path_output is not None:
            fig = Figure(figsize=(8, 10))
            ax1 = fig.add_subplot(311)
            ax1.plot(time_offsets, r2_total_list)
            ax1.set_xlabel("Time offset [ms]")
            ax1.set_ylabel("Average r2")
            ax1.set_title("R2 score for different time offsets")
            ax1.set_ylim([-0.1, 1.1])

            # 750 ms is chosen as the smoothing length
            window_length = round(750 / ((max_bound_offset - min_bound_offset) / n_samples))
            r2_list_smooth = savgol_filter(r2_total_list, window_length, 4, mode='mirror')
            # The distance parameter is the minimum number of samples between adjacent peaks
            # 1000 ms is chosen as the minimum time between peaks
            min_distance = round(500 / ((max_bound_offset - min_bound_offset) / n_samples))
            peak_indices = find_peaks(r2_list_smooth, distance=min_distance, height=0.4)[0]
            for index in peak_indices:
                ax1.vlines(time_offsets[index], -1, 2, colors='k', linestyles='dashed')
                ax1.annotate(f"{round(time_offsets[index])}ms",
                             (time_offsets[index] + 50, r2_total_list[index] - 0.2))

            self.pmu.adjust_start_time(best_time_offset)

            acq_times = get_acquisition_times(self.nii_fieldmap_orig, self.json_fmap)
            pmu_plot_times = self.pmu.get_times(acq_times.min() - 1000, acq_times.max() + 1000)
            pmu_plot_pressures = (self.pmu.get_resp_trace(acq_times.min() - 1000, acq_times.max() + 1000) - 2048) / 100

            ax2 = fig.add_subplot(312)
            ax2.plot((pmu_plot_times - pmu_plot_times.min()) / 1000, pmu_plot_pressures, label='pmu')
            for i_slice in range(n_slices):
                y = fmap_ma.mean(axis=(0, 1))[i_slice].filled()
                y = (y - y.mean())
                ax2.scatter((acq_times[:, i_slice] - pmu_plot_times.min()) / 1000, y, label=f"slice: {i_slice}")

            ax2.legend()
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Field [Hz]")
            ax2.set_title(f"B0 offset and acquired pressure through time with time offset: {round(best_time_offset)}ms")

            ax3 = fig.add_subplot(313)
            for i_slice in range(n_slices):
                if i_slice in []:
                    continue
                pressures = self.pmu.interp_resp_trace(acq_times) - 2048
                y = fmap_ma.mean(axis=(0, 1))[i_slice].filled()
                y = (y - y.mean())
                reg = LinearRegression().fit(pressures[:, i_slice].reshape(-1, 1), y)
                # Adjusted r2 score
                r2 = reg.score(pressures[:, i_slice].reshape(-1, 1), y)
                r2_corr = (1 - (1 - r2) * (len(y) - 1) / (len(y) - pressures[:, i_slice].reshape(-1, 1).shape[1] - 1))
                ax3.scatter(pressures[:, i_slice], y, label=f"slice: {i_slice}, score: {r2:.2}")
                ax3.plot(pressures[:, i_slice], reg.predict(pressures[:, i_slice].reshape(-1, 1)))

            ax3.set_xlabel("Pressure [-2048,2048]")
            ax3.set_ylabel("Field [Hz]")
            ax3.legend()
            ax3.set_title(f"B0 offset vs pressure with time offset: {round(best_time_offset)}ms")

            fname_figure = os.path.join(self.path_output, 'fig_rt_pmu_offset_scan.png')
            fig.tight_layout()
            fig.savefig(fname_figure, bbox_inches='tight')
            logger.debug(f"Saved figure: {fname_figure}")

        self.pmu.adjust_start_time(previous_time_offset)

        return round(best_time_offset)

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
        extended_fmap = False
        for i_axis in range(3):
            if nii_fmap_orig.shape[i_axis] < self.mask_dilation_kernel_size:
                nii_fieldmap, location = extend_fmap_to_kernel_size(nii_fmap_orig, self.mask_dilation_kernel_size,
                                                                    self.path_output, ret_location=True)
                extended_fmap = True
                self.fmap_orig_location = location
                break

        return nii_fieldmap, nii_fmap_orig, extended_fmap

    def load_masks(self, nii_static_mask, nii_riro_mask):
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
        if not np.all(nii_static_mask.shape == anat.shape) or not np.all(nii_static_mask.affine == self.nii_anat.affine):
            # Resample the static mask on the target anatomical image
            logger.debug("Resampling static mask on the target anat")
            nii_static_mask = resample_from_to(nii_static_mask, self.nii_anat, order=1, mode='grid-constant')
            # Save the resampled mask
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_static_mask, os.path.join(self.path_output, "mask_static_resampled_on_anat.nii.gz"))

        if not np.all(nii_riro_mask.shape == anat.shape) or not np.all(nii_riro_mask.affine == self.nii_anat.affine):
            # Resample the riro mask on the target anatomical image
            logger.debug("Resampling riro mask on the target anat")
            nii_riro_mask = resample_from_to(nii_riro_mask, self.nii_anat, order=1, mode='grid-constant')
            # Save the resampled mask
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_riro_mask, os.path.join(self.path_output, "mask_riro_resampled_on_anat.nii.gz"))

        return nii_static_mask, nii_riro_mask

    def get_acq_pressures(self):
        """
        Get the acquisition pressures at the times when the field map volumes and slices were acquired.

        Returns:
            numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
        """
        # Fetch PMU timing
        self.acq_timestamps_orig = get_acquisition_times(self.nii_fieldmap_orig, self.json_fmap)
        if self.extended_fmap:
            # If the field map was extended, we need to add extra slices to the acq_timestamps
            n_slices_to_extend = int((self.nii_fieldmap.shape[2] - self.acq_timestamps_orig.shape[1]) / 2)
            self.acq_timestamps = np.zeros((self.acq_timestamps_orig.shape[0], self.nii_fieldmap.shape[2]))
            for i_slice_to_extend in range(n_slices_to_extend):
                self.acq_timestamps[:, i_slice_to_extend] = self.acq_timestamps_orig[:, 0]
                self.acq_timestamps[:, -i_slice_to_extend - 1] = self.acq_timestamps_orig[:, -1]
            self.acq_timestamps[:, n_slices_to_extend:-n_slices_to_extend] = self.acq_timestamps_orig
        else:
            self.acq_timestamps = self.acq_timestamps_orig

        # TODO: deal with saturation
        # fit PMU and fieldmap values
        self.acq_pressures_orig = self.pmu.interp_resp_trace(self.acq_timestamps_orig)
        self.acq_pressures = self.pmu.interp_resp_trace(self.acq_timestamps)

    def get_real_time_parameters(self):
        """
        Get real time parameters used for shimming

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 3D array containing the static data for the optimization
                * np.ndarray: 3D array containing the real time data for the optimization
                * float: Mean pressure of the respiratory trace.
                * float: Root mean squared of the pressure trace. This is provided to compare results between scans,
                         multiply the riro coefficients by rms of the pressure to do so.

        """
        fieldmap = self.nii_fieldmap.get_fdata()

        n_slices = fieldmap.shape[2]
        n_volumes = fieldmap.shape[-1]

        # regularization --> static, riro
        # field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
        mean_p = self.pmu.mean(self.acq_timestamps[0].min(), self.acq_timestamps[-1].max())
        pressure_rms = self.pmu.get_pressure_rms(self.acq_timestamps[0].min(), self.acq_timestamps[-1].max())

        # Mask the voxels not being shimmed for riro
        mask_fmap = np.zeros_like(self.mask_static_fmcs_dil)
        mask_fmap[self.mask_riro_fmcs_dil != 0] = self.mask_static_fmcs_dil[self.mask_riro_fmcs_dil != 0]
        masked_fieldmap = np.repeat(mask_fmap[..., np.newaxis], fieldmap.shape[-1], 3) * fieldmap

        static = np.zeros(fieldmap.shape[:-1])
        riro = np.zeros(fieldmap.shape[:-1])

        for i_slice in range(n_slices):
            x = self.acq_pressures[:, i_slice].reshape(-1, 1) - mean_p

            # Safety check for linear regression if the pressure and field map fit well
            y = masked_fieldmap[..., i_slice, :].reshape(-1, n_volumes).T

            reg_riro = LinearRegression().fit(x, y)
            # TODO: There are a lot of 0s in there (it is masked) so the score is biased
            # Calculate adjusted r2 score (Takes into account the number of observations and predictor variables)
            score_riro = 1 - (1 - reg_riro.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
            logger.debug(
                f"Linear fit of the RIRO masked for slice: {i_slice} fieldmap and pressure"
                f"got a R2 score of: {score_riro}")

            # Warn if lower than a threshold
            # Threshold was set by looking at a small sample of data (This value could be updated based on user
            # feedback)
            threshold_score = 0.7
            if score_riro < threshold_score:
                logger.warning(
                    f"Linear fit of the RIRO masked fieldmap for slice {i_slice} and pressure got a low R2"
                    f"score: {score_riro} (less than {threshold_score}). This indicates a bad fit between the pressure"
                    f"data and the fieldmap values")

            # Fit to the linear model (no mask)
            y = fieldmap[..., i_slice, :].reshape(-1, n_volumes).T
            reg = LinearRegression().fit(x, y)

            # static/riro contains a 3d matrix of static/riro map in the fieldmap space considering the previous equation
            static[..., i_slice] = reg.intercept_.reshape(fieldmap.shape[:-2])
            riro[..., i_slice] = reg.coef_.reshape(
                fieldmap.shape[:-2])  # [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

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
        if self.method == 'bfgs':
            self.method = 'bfgs_rt'
        self.select_optimizer(riro, affine_fieldmap, self.pmu, mean_p=mean_p)

        # Static shim
        logger.info("Static optimization")
        coef_static = self.optimize(self.mask_static_fmcs_per_shim_dil)

        # RIRO optimization
        # Use the currents to define a list of new coil bounds for the riro optimization
        self.bounds = new_bounds_from_currents_static_to_riro(
            coef_static, self.optimizer.merged_bounds, self.coils_static, self.coils_riro)

        logger.info("Realtime optimization")
        coef_riro = self.optimize_riro(self.mask_riro_fmcs_per_shim_dil)

        # Multiplying by the RMS of the pressure allows to make abstraction of the tightness of the bellow
        # between scans. This allows to compare results between scans.
        # coef_riro_rms = coef_riro * pressure_rms
        # [unit_shim/unit_pressure] * rms_pressure, ex: [Hz/unit_pressure] * rms_pressure

        return coef_static, coef_riro, mean_p, pressure_rms

    def select_optimizer(self, unshimmed, affine, pmu: PmuResp = None, mean_p=None):
        """
        Select and initialize the optimizer

        Args:
            unshimmed (np.ndarray): 3D B0 map
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            pmu (PmuResp): PmuResp object containing the respiratory trace information. Required for method
                           'least_squares_rt'.
            mean_p (float): Mean pressure of the respiratory trace. Required for methods 'XXX_rt'.

        """

        # global supported_optimizers
        if self.method in supported_optimizers:
            if self.method in ['least_squares', 'bfgs']:
                self.optimizer = supported_optimizers[self.method](
                    self.coils_static, unshimmed, affine,
                                                                   self.opt_criteria, reg_factor=self.reg_factor)
            elif self.method == 'quad_prog':
                self.optimizer = supported_optimizers[self.method](self.coils_static, unshimmed, affine,
                                                                   reg_factor=self.reg_factor)

            elif self.method in ['least_squares_rt', 'bfgs_rt']:
                # Make sure pmu is defined
                if pmu is None:
                    raise ValueError(f"pmu parameter is required if using the optimization method: {self.method}")
                if mean_p is None:
                    raise ValueError(f"mean_p parameter is required if using the optimization method: {self.method}")

                # Add pmu to the realtime optimizer(s)
                self.optimizer_riro = supported_optimizers[self.method](self.coils_riro, unshimmed, affine,
                                                                        self.opt_criteria, pmu,
                                                                        mean_p=mean_p,
                                                                        reg_factor=self.reg_factor)
            elif self.method == 'quad_prog_rt':
                # Make sure pmu is defined
                if pmu is None:
                    raise ValueError(f"pmu parameter is required if using the optimization method: {self.method}")

                # Add pmu to the realtime optimizer(s)
                self.optimizer_riro = supported_optimizers[self.method](self.coils_riro, unshimmed, affine, pmu,
                                                                        reg_factor=self.reg_factor)

            else:
                if pmu is None:
                    self.optimizer = supported_optimizers[self.method](self.coils_static, unshimmed, affine)
                else:
                    self.optimizer_riro = supported_optimizers[self.method](self.coils_riro, unshimmed, affine)

        else:
            raise KeyError(f"Method: {self.method} is not part of the supported optimizers")

    def resample_masks_to_target_per_shim(self, nii_fmap):
        """
        Resample the static and riro masks to the target coordinate system for each shim group

        nii_target (nib.Nifti1Image): 4d fieldmap

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Static mask resampled on the fieldmap
                * np.ndarray: Riro mask resampled on the original fieldmap
                * np.ndarray: Static mask resampled and dilated on the fieldmap
                * np.ndarray: Riro mask resampled and dilated on the original fieldmap
        """
        n_shims = len(self.slices)

        nii_fmap_cs = nib.Nifti1Image(nii_fmap.get_fdata()[..., 0], nii_fmap.affine)

        r = Parallel(-1, backend='loky')(
            delayed(resample_mask)(self.nii_static_mask, nii_fmap_cs, self.slices[i],
                                   self.mask_dilation_kernel, self.mask_dilation_kernel_size,
                                   self.path_output, return_non_dil_mask=True)
            for i in range(n_shims))
        static_mask, static_mask_dil = zip(*r)
        r = Parallel(-1, backend='loky')(
            delayed(resample_mask)(self.nii_riro_mask, nii_fmap_cs, self.slices[i],
                                   self.mask_dilation_kernel,
                                   self.mask_dilation_kernel_size, self.path_output, return_non_dil_mask=True)
            for i in range(n_shims))
        riro_mask, riro_mask_dil = zip(*r)

        static_mask_fmap_cs_per_shim = np.array(
            [static_mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
        static_mask_fmap_cs_per_shim_dil = np.array(
            [static_mask_dil[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
        riro_mask_fmap_cs_per_shim = np.array(
            [riro_mask[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)
        riro_mask_fmap_cs_per_shim_dil = np.array(
            [riro_mask_dil[it].get_fdata() for it in range(n_shims)]).transpose(1, 2, 3, 0)

        return (static_mask_fmap_cs_per_shim,
                riro_mask_fmap_cs_per_shim,
                static_mask_fmap_cs_per_shim_dil,
                riro_mask_fmap_cs_per_shim_dil)

    def resample_mask_to_target(self, nii_target):
        """
        Resample the static and riro masks to the target coordinate system

        Args:
            nii_target (nib.Nifti1Image): 4d fieldmap

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: Static mask resampled on the fieldmap
                * np.ndarray: Riro mask resampled on the original fieldmap
                * np.ndarray: Static mask resampled and dilated on the fieldmap
                * np.ndarray: Riro mask resampled and dilated on the original fieldmap
        """
        nii_3dfmap = nib.Nifti1Image(nii_target.get_fdata()[..., 0], nii_target.affine,
                                     header=nii_target.header)
        fmap_mask_static, fmap_mask_static_dil = resample_mask(self.nii_static_mask, nii_3dfmap,
                                                               tuple(range(self.nii_anat.shape[2])),
                                                               dilation_kernel=self.mask_dilation_kernel,
                                                               dilation_size=self.mask_dilation_kernel_size,
                                                               return_non_dil_mask=True)
        fmap_mask_riro, fmap_mask_riro_dil = resample_mask(self.nii_riro_mask, nii_3dfmap,
                                                           tuple(range(self.nii_anat.shape[2])),
                                                           dilation_kernel=self.mask_dilation_kernel,
                                                           dilation_size=self.mask_dilation_kernel_size,
                                                           return_non_dil_mask=True)

        return (fmap_mask_static.get_fdata(),
                fmap_mask_riro.get_fdata(),
                fmap_mask_static_dil.get_fdata(),
                fmap_mask_riro_dil.get_fdata())

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
        unshimmed = self.nii_fieldmap_orig.get_fdata()
        shape = unshimmed.shape + (len(self.slices),)
        shimmed_static_riro = np.zeros(shape)
        shimmed_static = np.zeros(shape)
        shimmed_riro = np.zeros(shape)
        masked_shim_static_riro = np.zeros(shape)
        masked_unshimmed = np.zeros(shape)
        shim_trace_static_riro = []
        shim_trace_static = []
        shim_trace_riro = []
        unshimmed_trace = []

        # Combine static and riro masks
        mask_fmcs_per_shim = np.maximum(self.mask_static_orig_fmcs_per_shim, self.mask_riro_orig_fmcs_per_shim)
        mask_fmcs = np.maximum(self.mask_static_orig_fmcs, self.mask_riro_orig_fmcs)

        if self.extended_fmap:
            # Remove extended slices if the field map was smaller than the kernel size
            n_channels = self.optimizer.merged_coils.shape[-1]
            # static coil
            merged_coils = self.optimizer.merged_coils[self.fmap_orig_location[..., 0], :]
            merged_coils = merged_coils.reshape(unshimmed.shape[:-1] + (n_channels,))
            # riro coil
            merged_coils_riro = self.optimizer_riro.merged_coils[self.fmap_orig_location[..., 0], :]
            merged_coils_riro = merged_coils_riro.reshape(unshimmed.shape[:-1] + (n_channels,))
        else:
            merged_coils = self.optimizer.merged_coils
            merged_coils_riro = self.optimizer_riro.merged_coils

        for i_shim in range(len(self.slices)):
            # Calculate static correction
            correction_static = merged_coils @ coef_static[i_shim]

            # Calculate the riro coil profiles
            riro_profile = merged_coils_riro @ coef_riro[i_shim]

            for i_t in range(self.nii_fieldmap.shape[3]):
                # Apply the static and riro correction
                correction_riro = riro_profile * (self.acq_pressures_orig[i_t] - mean_p)
                shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
                shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
                shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

                # Calculate masked shim
                mask_fmcs_per_shim_bin = (mask_fmcs_per_shim != 0).astype(int)
                masked_shim_static_riro[..., i_t, i_shim] = (mask_fmcs_per_shim_bin[..., i_shim] * shimmed_static_riro[..., i_t, i_shim])
                masked_unshimmed[..., i_t, i_shim] = mask_fmcs_per_shim_bin[..., i_shim] * unshimmed[..., i_t]

                # Calculate weighted RMSE
                # TODO: Calculate the sum of mask_fmap_cs[..., i_shim] and divide by that (If the roi is bigger due to
                #  interpolation, it should not count more). Possibly use soft mask?
                rmse_shimmed_static = calculate_metric_within_mask(shimmed_static[..., i_t, i_shim],
                                                                   mask_fmcs_per_shim[..., i_shim],
                                                                   metric='rmse')
                rmse_shimmed_static_riro = calculate_metric_within_mask(shimmed_static_riro[..., i_t, i_shim],
                                                                        mask_fmcs_per_shim[..., i_shim],
                                                                        metric='rmse')
                rmse_shimmed_riro = calculate_metric_within_mask(shimmed_riro[..., i_t, i_shim],
                                                                 mask_fmcs_per_shim[..., i_shim],
                                                                 metric='rmse')
                rmse_unshimmed = calculate_metric_within_mask(unshimmed[..., i_t],
                                                              mask_fmcs_per_shim[..., i_shim],
                                                              metric='rmse')

                if rmse_shimmed_static_riro > rmse_unshimmed:
                    logger.warning("Verify the shim parameters. Some give worse results than no shim.\n"
                                   f"i_shim: {i_shim}, i_t: {i_t}")

                riro_current_txt = ""
                for i_fmap_slice in range(unshimmed.shape[2]):
                    riro_current_txt += f"Fmap slice {i_fmap_slice}: {coef_riro[i_shim] * (self.acq_pressures_orig[i_t][i_fmap_slice] - mean_p)}\n"
                logger.debug(f"\nRMSE: i_shim: {i_shim}, t: {i_t}"
                             f"\nunshimmed: {rmse_unshimmed}, shimmed static: {rmse_shimmed_static}, "
                             f"shimmed static+riro: {rmse_shimmed_static_riro}\n"
                             f"Static currents:\n{coef_static[i_shim]}\n"
                             f"Riro currents:\n" + riro_current_txt)

                # Create a 1D list of the sum of the shimmed and unshimmed maps
                shim_trace_static.append(rmse_shimmed_static)
                shim_trace_static_riro.append(rmse_shimmed_static_riro)
                shim_trace_riro.append(rmse_shimmed_riro)
                unshimmed_trace.append(rmse_unshimmed)

        # reshape to slice x timepoint
        nt = unshimmed.shape[3]
        n_shim = len(self.slices)
        shim_trace_static = np.array(shim_trace_static).reshape(n_shim, nt)
        shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_shim, nt)
        shim_trace_riro = np.array(shim_trace_riro).reshape(n_shim, nt)
        unshimmed_trace = np.array(unshimmed_trace).reshape(n_shim, nt)

        if self.path_output is not None:
            # Plot before vs after shimming averaged on time
            shimmed_mask_avg = np.zeros(mask_fmcs.shape)
            np.divide(np.sum(np.mean(masked_shim_static_riro, axis=3), axis=3), np.sum(mask_fmcs_per_shim, axis=3),
                      where=mask_fmcs.astype(bool), out=shimmed_mask_avg)
            plot_full_mask(np.mean(unshimmed, axis=3), shimmed_mask_avg, mask_fmcs, self.path_output)

            # Plot STD over time before and after shimming
            self.plot_full_time_std(unshimmed, masked_shim_static_riro, mask_fmcs_per_shim, mask_fmcs)

        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            # plot results
            self.plot_currents(coef_static, riro=coef_riro * pressure_rms)
            self.plot_shimmed_trace(unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro)
            self.plot_pressure_and_unshimmed_field(unshimmed_trace)
            self.plot_pressure_vs_field(masked_unshimmed, mask_fmcs_per_shim)
            self.print_rt_metrics(unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro, mask_fmcs_per_shim)
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

        # Cannot see the evolution of the currents through shims if there is only one
        if len(self.slices) == 1:
            return

        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        n_channels = static.shape[1]
        for i_channel in range(n_channels):
            ax.plot(static[:, i_channel], label=f"Static channel {i_channel} currents through shim groups")

        if riro is not None:
            n_channels = riro.shape[1]
            for i_channel in range(n_channels):
                ax.plot(riro[:, i_channel], label=f"Riro channel {i_channel} currents through shim groups")

        ax.set_xlabel('Shim group')
        ax.set_ylabel('Coefficients')
        ax.legend()
        ax.set_title("Currents through shims")
        fname_figure = os.path.join(self.path_output, 'fig_currents.png')
        fig.savefig(fname_figure)
        logger.debug(f"Saved figure: {fname_figure}")

    def plot_pressure_vs_field(self, unshimmed, mask_fm):
        """ One graph per i_shim
        In each graph, one scatter and one line for each fmap slice in the ROI
        Each line should have pearson correlation coefficient
        """
        # x, y, z, t, i_shim

        n_t = unshimmed.shape[3]
        n_shims = len(self.slices)
        n_slices_fm = unshimmed.shape[2]

        # Binarize mask
        mask_fm = (mask_fm != 0).astype(int)

        # Remove
        plots = []
        for i_shim in range(n_shims):
            if np.any(mask_fm[..., i_shim] != 0):
                plots.append(i_shim)

        path_pressure_and_unshimmed_field = os.path.join(self.path_output, 'fig_noshim_vs_pressure_regression')
        create_output_dir(path_pressure_and_unshimmed_field)

        for i_plot, i_shim in enumerate(plots):
            fm_slices = []
            for i_slice_fm in range(n_slices_fm):
                if np.any(mask_fm[..., i_slice_fm, i_shim] != 0):
                    fm_slices.append(i_slice_fm)

            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            y = np.zeros((n_t, len(fm_slices)))
            # pressure
            for i, i_slice_fm in enumerate(fm_slices):
                x = self.acq_pressures[:, i_slice_fm]

                for i_t in range(n_t):
                    y[i_t, i] = calculate_metric_within_mask(unshimmed[..., i_slice_fm, i_t, i_shim],
                                                             mask_fm[..., i_slice_fm, i_shim],
                                                             metric='rmse')

                reg = LinearRegression().fit(x.reshape(-1, 1), y[:, i])
                # Adjusted r2 score
                score = (1 - (1 - reg.score(x.reshape(-1, 1), y[:, i])) *
                         (len(y[:, i]) - 1) / (len(y[:, i]) - x.reshape(-1, 1).shape[1] - 1))

                ax.scatter(x, y[:, i], label=f"Fm slice: {i_slice_fm}, r2: {score:.2f}")
                ax.plot(x, reg.predict(x.reshape(-1, 1)))

            # If there is only 1 fm slice, it's the same as all the slices
            if len(fm_slices) != 1:
                x = self.acq_pressures[:, fm_slices]
                reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1))
                # Adjusted r2 score
                score = (1 - (1 - reg.score(x.reshape(-1, 1), y.reshape(-1))) *
                         (len(y.reshape(-1)) - 1) / (len(y.reshape(-1)) - x.reshape(-1, 1).shape[1] - 1))
                ax.plot(x.reshape(-1), reg.predict(x.reshape(-1, 1)),
                        label=f"All slices: {i_slice_fm}, r2: {score:.2f}")

            ax.legend()
            ax.set_xlabel('Pressure (A.U.)')
            ax.set_ylabel('RMSE (Hz)')
            ax.set_title(f"Pressure vs Field for target slice(s): {self.slices[i_shim]}")
            fname_figure = os.path.join(path_pressure_and_unshimmed_field,
                                        f'fig_noshim_vs_pressure_regression_shimgroup_{i_shim:03}.png')
            fig.savefig(fname_figure, bbox_inches='tight')

    def plot_pressure_and_unshimmed_field(self, unshimmed_trace):
        """
        Plot respiratory trace, acquisition time pressure points and the B0 field RMSE

        Args:
            unshimmed_trace (np.ndarray): field in the ROI for each shim volume
        """
        # Get the pmu data values in the range of the acquisition
        pmu_timestamps = self.pmu.get_times(self.acq_timestamps[0].min() - 1000, self.acq_timestamps[-1].max() + 1000)
        pmu_pressures = self.pmu.get_resp_trace(self.acq_timestamps[0].min() - 1000,
                                                self.acq_timestamps[-1].max() + 1000)

        # Select slices shimmed
        curated_unshimmed_trace = unshimmed_trace[self.index_shimmed]

        # Get the b0 field in the same units as the pressure reading
        n_plots = len(self.index_shimmed)

        max_diff_field_list = max(curated_unshimmed_trace, key=lambda x: abs(x.max() - x.min()))
        min_field = max_diff_field_list.min()
        max_field = max_diff_field_list.max()
        max_diff_field = max_field - min_field

        diff_pressure = pmu_pressures.max() - pmu_pressures.min()
        scaling = max_diff_field / diff_pressure
        avg_pressure = np.mean(pmu_pressures)

        # Scale
        curated_unshimmed_trace_scaled = np.array([(x - np.mean(x)) / scaling + avg_pressure
                                                   for x in curated_unshimmed_trace])

        # Find y limits
        perc = (self.pmu.max - self.pmu.min) / 20
        ylim = (min(curated_unshimmed_trace_scaled.min(), self.pmu.min - perc),
                max(curated_unshimmed_trace_scaled.max(), self.pmu.max + perc))

        # Plot
        path_pressure_and_unshimmed_field = os.path.join(self.path_output, 'fig_noshim_vs_pressure')
        create_output_dir(path_pressure_and_unshimmed_field)

        for i_plot in range(n_plots):
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.plot((pmu_timestamps - pmu_timestamps[0]) / 1000, pmu_pressures,
                    label='Pressure Trace')
            ax.plot((self.acq_timestamps_orig - pmu_timestamps[0]).mean(axis=1) / 1000,
                    curated_unshimmed_trace_scaled[i_plot],
                    label='Unshimmed RMSE over the ROI')
            ax.scatter((np.mean(self.acq_timestamps_orig, axis=1) - pmu_timestamps[0]) / 1000,
                       np.mean(self.acq_pressures_orig, axis=1),
                       color='red',
                       label='Field map timepoints')
            ax.legend()
            ax.set_ylim(ylim)
            ax.set_yticks([pmu_pressures.min(), pmu_pressures.max()],
                          [min_field.astype(int), max_field.astype(int)])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RMSE (Hz)')
            ax.set_title(f"Slices: {self.slices[self.index_shimmed[i_plot]]}")

            # Save figure
            fname_figure = os.path.join(path_pressure_and_unshimmed_field,
                                        f'fig_noshim_vs_pressure_shimgroup_{self.index_shimmed[i_plot]:03}.png')
            fig.savefig(fname_figure, bbox_inches='tight')

        logger.debug(f"Saved figures: {path_pressure_and_unshimmed_field}")

    def plot_shimmed_trace(self, unshimmed_trace, shim_trace_static, shim_trace_riro, shim_trace_static_riro):
        """
        Plot shimmed and unshimmed rmse over the roi for each shim

        Args:
            unshimmed_trace (np.ndarray): array with the trace of the nii_fieldmap data
            shim_trace_static (np.ndarray): array with the trace of the nii_fieldmap data after the static shimming
            shim_trace_riro (np.ndarray): array with the trace of the nii_fieldmap data after the riro shimming
            shim_trace_static_riro (np.ndarray): array with the trace of the nii_fieldmap data after both shimming
        """

        min_value = min(
            shim_trace_static_riro[self.index_shimmed, :].min(),
            shim_trace_static[self.index_shimmed, :].min(),
            shim_trace_riro[self.index_shimmed, :].min(),
            unshimmed_trace[self.index_shimmed, :].min()
        )
        max_value = max(
            shim_trace_static_riro[self.index_shimmed, :].max(),
            shim_trace_static[self.index_shimmed, :].max(),
            shim_trace_riro[self.index_shimmed, :].max(),
            unshimmed_trace[self.index_shimmed, :].max()
        )

        path_shimmed_trace = os.path.join(self.path_output, 'fig_trace_shimmed_vs_unshimmed')
        create_output_dir(path_shimmed_trace)

        # Calc ysize
        for i, i_shim in enumerate(self.index_shimmed):
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(shim_trace_static_riro[i_shim, :], label='shimmed static + riro')
            ax.plot(shim_trace_static[i_shim, :], label='shimmed static')
            ax.plot(shim_trace_riro[i_shim, :], label='shimmed_riro')
            ax.plot(unshimmed_trace[i_shim, :], label='unshimmed')
            ax.set_xlabel('Timepoints')
            ax.set_ylabel('RMSE over the ROI (Hz)')
            ax.legend()
            ax.set_ylim([min_value, max_value])
            ax.set_title(f"Unshimmed vs shimmed values: shim {self.slices[i_shim]}")
            fname_figure = os.path.join(path_shimmed_trace, f'fig_trace_shimmed_vs_unshimmed_shimgroup_{i_shim:03}.png')
            fig.savefig(fname_figure, bbox_inches='tight')
        logger.debug(f"Saved figures: {path_shimmed_trace}")

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
        ma_unshimmed = np.ma.array(unshimmed_repeat, mask=mask_repeats==0)
        ma_shim_static = np.ma.array(shimmed_static, mask=mask_repeats==0)
        ma_shim_static_riro = np.ma.array(shimmed_static_riro, mask=mask_repeats==0)
        ma_shim_riro = np.ma.array(shimmed_riro, mask=mask_repeats==0)

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

    def plot_full_time_std(self, unshimmed, masked_shim_static_riro, mask_fmap_cs, mask):
        """
        Plot and save the std heatmap over time

        Args:
            unshimmed (np.ndarray): Original fieldmap not shimmed shaped (x, y, z, time)
            masked_shim_static_riro(np.ndarray): Masked shimmed fieldmap shaped (x, y, z, time, slices)
            mask_fmap_cs (np.ndarray): Field map mask indicating where delta B0 is not 0 in each slice -- shaped (x, y, z, slices)
            mask (np.ndarray): Mask in the fieldmap space shaped (x, y, z)
        """
        # Transform shimmed field map to shape (x, y, z, time)
        sum_mask_fmap_cs = np.sum(mask_fmap_cs, axis=3)
        mask_extended = np.repeat(mask[..., np.newaxis], masked_shim_static_riro.shape[-2], axis=-1)

        # Transpose is used to cater to numpy division order
        # (3, 2, 4) / (3, 2) Does not work
        # (4, 2, 3) / (2, 3) Does work
        # * Using out parameter in np.divide() prevents inconsistent results
        shimmed_masked = np.zeros(mask_extended.shape)
        np.divide(np.sum(masked_shim_static_riro, axis=-1).T, sum_mask_fmap_cs.T, where=mask.T.astype(bool),
                  out=shimmed_masked.T)

        std_shimmed_masked = np.std(shimmed_masked, axis=-1, dtype=np.float64)
        std_unshimmed = np.std(unshimmed, axis=-1, dtype=np.float64)

        # Plot
        nan_unshimmed_masked = np.ma.array(std_unshimmed, mask=(mask==0), fill_value=np.nan)
        nan_shimmed_masked = np.ma.array(std_shimmed_masked, mask=(mask==0), fill_value=np.nan)

        mt_unshimmed = montage(np.mean(unshimmed, axis=-1))
        mt_unshimmed_masked = montage(nan_unshimmed_masked.filled())
        mt_shimmed_masked = montage(nan_shimmed_masked.filled())

        # Compute weighted mean
        metric_unshimmed_mean = calculate_metric_within_mask(std_unshimmed, mask, metric='mean')
        metric_shimmed_mean = calculate_metric_within_mask(std_shimmed_masked, mask, metric='mean')

        # Remove the outliers to calculate the colorbar limits
        # Necessary because some STD are much higher and are not visible on the heatmap, they are still considered in
        # the metric
        shim_limit = np.nanpercentile(mt_shimmed_masked[mt_shimmed_masked != 0], 90)
        unshim_limit = np.nanpercentile(mt_unshimmed_masked[mt_unshimmed_masked != 0], 90)

        min_value = min(np.nanmin(mt_unshimmed_masked), np.nanmin(mt_shimmed_masked))
        max_value = max(np.nanmax(mt_unshimmed_masked[mt_unshimmed_masked < unshim_limit]),
                        np.nanmax(mt_shimmed_masked[mt_shimmed_masked < shim_limit]))

        fig = Figure(figsize=(9, 6))
        fig.suptitle("Fieldmaps\nFieldmap Coordinate System\n\u0394B\u2080 STD over time ")

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(mt_unshimmed, cmap='gray')
        mt_unshimmed_masked[mt_unshimmed_masked == 0] = np.nan
        im = ax.imshow(mt_unshimmed_masked, vmin=min_value, vmax=max_value, cmap='jet')
        ax.set_title(f"Before shimming\nmean: {metric_unshimmed_mean:.3}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(mt_unshimmed, cmap='gray')
        mt_shimmed_masked[mt_shimmed_masked == 0] = np.nan
        im = ax.imshow(mt_shimmed_masked, vmin=min_value, vmax=max_value, cmap='jet')
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


def plot_full_mask(unshimmed, shimmed_masked, mask, path_output):
    """
    Plot and save the static full mask

    Args:
        unshimmed (np.ndarray): Original fieldmap not shimmed
        shimmed_masked(np.ndarray): Masked shimmed fieldmap
        mask (np.ndarray): Mask in the fieldmap space
        path_output (str): Path to the output folder
    """

    # Get the binary mask from the soft mask
    bin_mask = (mask != 0).astype(int)

    # Plot
    nan_unshimmed_masked = np.ma.array(unshimmed, mask=(bin_mask==0), fill_value=np.nan)
    nan_shimmed_masked = np.ma.array(shimmed_masked, mask=(bin_mask==0), fill_value=np.nan)

    mt_unshimmed = montage(unshimmed)
    mt_unshimmed_masked = montage(nan_unshimmed_masked.filled())
    mt_shimmed_masked = montage(nan_shimmed_masked.filled())

    metric_unshimmed_std = calculate_metric_within_mask(unshimmed, mask, metric='std')
    metric_shimmed_std = calculate_metric_within_mask(shimmed_masked, mask, metric='std')
    metric_unshimmed_mean = calculate_metric_within_mask(unshimmed, mask, metric='mean')
    metric_shimmed_mean = calculate_metric_within_mask(shimmed_masked, mask, metric='mean')
    metric_unshimmed_mae = calculate_metric_within_mask(unshimmed, mask, metric='mae')
    metric_shimmed_mae = calculate_metric_within_mask(shimmed_masked, mask, metric='mae')
    metric_unshimmed_rmse = calculate_metric_within_mask(unshimmed, mask, metric='rmse')
    metric_shimmed_rmse = calculate_metric_within_mask(shimmed_masked, mask, metric='rmse')

    min_value = -100
    max_value = 100

    # Create figure
    fig = Figure(figsize=(15, 9))
    fig.suptitle("Fieldmaps\nFieldmap Coordinate System")

    # LEFT PANEL – Before shimming
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(mt_unshimmed, cmap='gray')
    im = ax.imshow(mt_unshimmed_masked, vmin=min_value, vmax=max_value, cmap='bwr')
    ax.set_title(f"Before shimming\nstd: {metric_unshimmed_std:.1f}, mean: {metric_unshimmed_mean:.1f}\n"
                 f"mae: {metric_unshimmed_mae:.1f}, rmse: {metric_unshimmed_rmse:.1f}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    # RIGHT PANEL – After shimming
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mt_unshimmed, cmap='gray')
    im = ax.imshow(mt_shimmed_masked, vmin=min_value, vmax=max_value, cmap='bwr')
    ax.set_title(f"After shimming\nstd: {metric_shimmed_std:.1f}, mean: {metric_shimmed_mean:.1f}\n"
                 f"mae: {metric_shimmed_mae:.1f}, rmse: {metric_shimmed_rmse:.1f}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    # Save
    fname_figure = os.path.join(path_output, 'fig_shimmed_vs_unshimmed.png')
    fig.savefig(fname_figure, bbox_inches='tight')


def new_bounds_from_currents(currents: dict, old_bounds: dict):
    """
    Uses the currents to determine the appropriate bounds for the next optimization. It assumes that
    "old_coef + next_bound < old_bound".

    Args:
        currents (dict): Dictionary with n_shims as keys each with a list of n_channels values.
        old_bounds (dict): Dictionary with orders as keys containing (min, max) containing the merged bounds of the previous
                           optimization.
    Returns:
        dict: Modified bounds (same shape as old_bounds)
    """
    new_bounds = {}
    for key in old_bounds:
        new_bounds[key] = []
        for i, bound in enumerate(old_bounds[key]):
            if bound == [None, None]:
                new_bounds[key].append(bound)
            elif bound[0] is None:
                new_bounds[key].append([None, bound[1] - currents[key][i]])
            elif bound[1] is None:
                new_bounds[key].append([bound[0] - currents[key][i], None])
            else:
                new_bounds[key].append([bound[0] - currents[key][i], bound[1] - currents[key][i]])
    return new_bounds


def new_bounds_from_currents_static_to_riro(currents, old_bounds, coils_static=[], coils_riro=[]):
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

    currents_riro = np.empty((currents.shape[0], 0))
    old_bounds_riro = []
    static_coil_names = [c.name for c in coils_static]

    index = 0
    coil_indexes = {}
    for coil in coils_static:
        if type(coil) == Coil:
            coil_indexes[coil.name] = [index, index + len(coil.coef_channel_minmax['coil'])]
            index += len(coil.coef_channel_minmax['coil'])
        else:
            coil_indexes[coil.name] = {}
            for key in coil.coef_channel_minmax:
                coil_indexes[coil.name][key] = [index, index + len(coil.coef_channel_minmax[key])]
                index += len(coil.coef_channel_minmax[key])

    for i, coil in enumerate(coils_riro):
        if coil.name in static_coil_names:
            if type(coil) == Coil:
                currents_riro = np.append(currents_riro,
                                          currents[:, coil_indexes[coil.name][0]:coil_indexes[coil.name][1]],
                                          axis=1)
                old_bounds_riro.extend(old_bounds[coil_indexes[coil.name][0]:coil_indexes[coil.name][1]])
            else:
                for order in coil.coef_channel_minmax:
                    if order in coils_static[static_coil_names.index(coil.name)].coef_channel_minmax.keys():
                        currents_riro = np.append(
                            currents_riro,
                            currents[:,
                                                  coil_indexes[coil.name][order][0]: coil_indexes[coil.name][order][1]],
                            axis=1)
                        old_bounds_riro.extend(
                            old_bounds[coil_indexes[coil.name][order]
                                               [0]:coil_indexes[coil.name][order][1]])
                    else:
                        currents_riro = np.append(currents_riro,
                                                  np.zeros((currents.shape[0], len(coil.coef_channel_minmax[order]))),
                                                  axis=1)
                        old_bounds_riro.extend(coil.coef_channel_minmax[order])

        else:
            if type(coil) == Coil:
                currents_riro = np.append(currents_riro,
                                          np.zeros((currents.shape[0], len(coil.coef_channel_minmax['coil']))),
                                          axis=1)
                old_bounds_riro.extend(coil.coef_channel_minmax['coil'])

            else:
                for order in coil.coef_channel_minmax:
                    currents_riro = np.append(currents_riro,
                                              np.zeros((currents.shape[0], len(coil.coef_channel_minmax[order]))),
                                              axis=1)
                    old_bounds_riro.extend(coil.coef_channel_minmax[order])

    new_bounds = []
    for i_shim in range(currents_riro.shape[0]):
        shim_bound = []
        for i_channel in range(len(old_bounds_riro)):
            a_bound = old_bounds_riro[i_channel] - currents_riro[i_shim, i_channel]
            shim_bound.append(tuple(a_bound))
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
        raise RuntimeError("No tag SliceTiming to automatically parse slice data")

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
        shim_group = tuple(np.where(slice_timing == slice_timing[list_slices[0]])[0].astype(np.int32).tolist())
        # Add this as a tuple
        slices.append(shim_group)

        # Since the list_slices is sorted by slice_timing, the only similar values will be at the beginning
        n_to_remove = len(shim_group)
        list_slices = list_slices[n_to_remove:]

    return slices


def define_slices(n_slices: int, factor=1, method='ascending', software_version=None):
    """
    Define the slices to shim according to the output convention. (list of tuples)

    Args:
        n_slices (int): Number of total slices.
        factor (int): Number of slices per shim.
        method (str): Defines how the slices should be sorted, supported methods include: 'interleaved', 'ascending',
                      'descending', 'volume'. See Examples for more details.

    Returns:
        list: 1D list containing tuples of dim3 slices to shim. (dim1, dim2, dim3)

    Examples:
        ::
            slices = define_slices(10, 2, 'interleaved')
            print(slices)  # [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
            slices = define_slices(20, 5, 'ascending')
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

    if n_slices % factor != 0:
        raise ValueError("SMS method does not support leftover slices")

    if method == 'interleaved':
        if factor == 1:
            if n_slices % 2 == 0:
                range_1 = range(1, n_slices, 2)
                range_2 = range(0, n_slices, 2)
            else:
                range_1 = range(0, n_slices, 2)
                range_2 = range(1, n_slices, 2)

            for i_shim in range_1:
                slices.append((i_shim,))

            for i_shim in range_2:
                slices.append((i_shim,))

            leftover = n_slices % factor

        else:
            if software_version != 'syngo MR E11':
                logger.warning("SMS has only been tested with syngo MR E11. If you are using a different software "
                               "version, the slices might not be interleaved or grouped correctly.")

            if n_slices % 2 == 0:
                range_1 = range(1, n_shims, 2)
                range_2 = range(0, n_shims, 2)

            else:
                range_1 = range(0, n_shims, 2)
                range_2 = range(1, n_shims, 2)

            if n_slices // factor % 2 != 0:
                special_indexes = [i * n_shims for i in range(0, factor)]
                for i_shim in range_1:
                    slices.append(tuple([i_shim + special_index for special_index in special_indexes]))

                for i_shim in range_2:
                    slices.append(tuple([i_shim + special_index for special_index in special_indexes]))

            if n_slices // factor % 2 == 0:
                replace_index = n_shims // 2 // 2
                special_indexes = [i * n_shims for i in range(0, factor)]

                for i, i_shim in enumerate(range_1[:-1]):
                    if i == replace_index:
                        slices.append(tuple([range_1[-1] + special_index for special_index in special_indexes]))
                    slices.append(tuple([i_shim + special_index for special_index in special_indexes]))

                for i, i_shim in enumerate(range_2[1:]):
                    if i == replace_index:
                        slices.append(tuple([range_2[0] + special_index for special_index in special_indexes]))
                    slices.append(tuple([i_shim + special_index for special_index in special_indexes]))

    elif method == 'ascending':
        for i_shim in range(n_shims):
            slices.append(tuple(range(i_shim, n_shims * factor, n_shims)))

    elif method == 'descending':
        for i_shim in range(n_shims):
            slices.append(tuple(range(n_shims - i_shim - 1, n_slices, n_shims)))


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


def extend_fmap_to_kernel_size(nii_fmap_orig, dilation_kernel_size, path_output=None, ret_location=False):
    """
    Load the fmap and expand its dimensions to the kernel size

    Args:
        nii_fmap_orig (nib.Nifti1Image): 3d (dim1, dim2, dim3) or 4d (dim1, dim2, dim3, t) nii to be extended
        dilation_kernel_size: Size of the kernel
        path_output (str): Path to save the debug output
        ret_location (bool): If True, return the location of the original data in the new data
    Returns:
        nib.Nifti1Image: Nibabel object of the loaded and extended fieldmap
    """

    fieldmap_shape = nii_fmap_orig.shape[:3]

    # Extend the dimensions where the kernel is bigger than the number of voxels
    tmp_nii = copy.deepcopy(nii_fmap_orig)
    location = np.ones(nii_fmap_orig.shape)
    for i_axis in range(len(fieldmap_shape)):
        # If there are less voxels than the kernel size, extend in that axis
        if fieldmap_shape[i_axis] < dilation_kernel_size:
            diff = float(dilation_kernel_size - fieldmap_shape[i_axis])
            n_slices_to_extend = math.ceil(diff / 2)
            tmp_nii, location = extend_slice(tmp_nii, n_slices=n_slices_to_extend, axis=i_axis, location=location)

    nii_fmap = tmp_nii

    # If DEBUG, save the extended fieldmap
    if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
        fname_new_fmap = os.path.join(path_output, 'tmp_extended_fmap.nii.gz')
        nib.save(nii_fmap, fname_new_fmap)
        logger.debug(f"Extended fmap, saved the new fieldmap here: {fname_new_fmap}")

    if ret_location:
        return nii_fmap, location.astype(bool)

    return nii_fmap


def extend_slice(nii_array, n_slices=1, axis=2, location=None):
    """
    Adds n_slices on each side of the selected axis. It uses the nearest slice and copies it to fill the values.
    Updates the affine of the matrix to keep the input array in the same location.

    Args:
        nii_array (nib.Nifti1Image): 3d or 4d array to extend the dimensions along an axis.
        n_slices (int): Number of slices to add on each side of the selected axis.
        axis (int): Axis along which to insert the slice(s), Allowed axis: 0, 1, 2.
        location (np.array): Location where the original data is located in the new data.
    Returns:
        nib.Nifti1Image: Array extended with the appropriate affine to conserve where the original pixels were located.

    Examples:
        ::
            print(nii_array.get_fdata().shape)  # (50, 50, 1, 10)
            nii_out = extend_slice(nii_array, n_slices=1, axis=2)
            print(nii_out.get_fdata().shape)  # (50, 50, 3, 10)
    """
    # Locate original data in new data
    orig_data_in_new_data = location

    if nii_array.get_fdata().ndim == 3:
        extended = nii_array.get_fdata()
        extended = extended[..., np.newaxis]
        if location is not None:
            orig_data_in_new_data = orig_data_in_new_data[..., np.newaxis]
    elif nii_array.get_fdata().ndim == 4:
        extended = nii_array.get_fdata()
    else:
        raise ValueError("Unsupported number of dimensions for input array")

    for i_slice in range(n_slices):
        if axis == 0:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros(orig_data_in_new_data.shape[1:]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros(orig_data_in_new_data.shape[1:]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[-1, :, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[0, :, :, :], axis=axis)
        elif axis == 1:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros_like(orig_data_in_new_data[:, 0, :, :]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros_like(orig_data_in_new_data[:, 0, :, :]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[:, -1, :, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, 0, :, :], axis=axis)
        elif axis == 2:
            if location is not None:
                orig_data_in_new_data = np.insert(orig_data_in_new_data, -1,
                                                  np.zeros_like(orig_data_in_new_data[:, :, 0, :]),
                                                  axis=axis)
                orig_data_in_new_data = np.insert(orig_data_in_new_data, 0,
                                                  np.zeros_like(orig_data_in_new_data[:, :, 0, :]),
                                                  axis=axis)
            extended = np.insert(extended, -1, extended[:, :, -1, :], axis=axis)
            extended = np.insert(extended, 0, extended[:, :, 0, :], axis=axis)
        else:
            raise ValueError("Unsupported value for axis")

    new_affine = update_affine_for_ap_slices(nii_array.affine, n_slices, axis)

    if nii_array.get_fdata().ndim == 3:
        extended = extended[..., 0]

    nii_extended = nib.Nifti1Image(extended, new_affine, header=nii_array.header)

    if location is not None:
        return nii_extended, orig_data_in_new_data

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
