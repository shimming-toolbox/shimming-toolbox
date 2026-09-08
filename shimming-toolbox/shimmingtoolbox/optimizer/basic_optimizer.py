#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
import nibabel as nib
import scipy.linalg
import logging
from typing import List

from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.coordinates import resample_from_to
from shimmingtoolbox.masking.mask_utils import modify_binary_mask

ListCoil = List[Coil]
logger = logging.getLogger(__name__)


class Optimizer(object):
    """
    Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
    Use optimize(args) to optimize a given mask.
    For basic optimizer, uses *unbounded* pseudo-inverse.

    Attributes:
        coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
        unshimmed (np.ndarray): 3d array of unshimmed volume
        unshimmed_affine (np.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
        merged_coils (np.ndarray): 4d array containing all coil profiles resampled onto the target unshimmed array
                                      concatenated on the 4th dimension. See self.merge_coils() for more details.
        merged_bounds (list): list of bounds corresponding to each merged coils: merged_bounds[3] is the (min, max)
                              bound for merged_coils[..., 3]
        merged_onoff_channels (list): list of off channels for all channels in merged_coils
        mask_coefficients (np.ndarray): 1d array of coefficients corresponding to the mask used for optimization
        w_signal_loss (np.ndarray): 3d array of weights for signal loss
        w_signal_loss_xy (np.ndarray): 3d array of weights for signal loss in the x and y directions
        epi_te (float): Echo time for EPI sequence
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, reg_factor=0, w_signal_loss=None, w_signal_loss_xy=None, epi_te=None):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            reg_factor (float): Regularization factor for the optimization.
            w_signal_loss (float): Weight for the through-slice gradient minimization.
            w_signal_loss_xy (float): Weight for the in-plane gradient minimization.
            epi_te (float): Echo time for the EPI sequence. (ms)
        """
        # Logging
        self.logger = logging.getLogger()
        logging.basicConfig(filename='test_optimizer.log', filemode='w', level=logging.DEBUG)

        self.coils = coils
        self.unshimmed = np.array([])
        self.unshimmed_affine = []
        self.merged_coils = []
        self.merged_bounds = []
        self.merged_onoff_channels = []
        self.merged_bounds_off_channels = []
        self.mask_coefficients = None
        self.set_unshimmed(unshimmed, affine)
        self.reg_factor = reg_factor

        # Initialization of signal loss parameters
        self.w_signal_loss = w_signal_loss
        self.w_signal_loss_xy = w_signal_loss_xy
        self.epi_te = epi_te

        self.coil_Gx_mat = None
        self.coil_Gy_mat = None
        self.coil_Gz_mat = None
        self.unshimmed_Gx_vec = None
        self.unshimmed_Gy_vec = None
        self.unshimmed_Gz_vec = None

    def set_unshimmed(self, unshimmed, affine):
        """
        Set the unshimmed array to a new array. Resamples coil profiles accordingly.

        Args:
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the qform affine transformation for the unshimmed array
        """
        # Check dimensions of unshimmed map
        if unshimmed.ndim != 3:
            raise ValueError(f"Unshimmed profile has {unshimmed.ndim} dimensions, expected 3 (dim1, dim2, dim3)")

        # Check dimensions of affine
        if affine.shape != (4, 4):
            raise ValueError("Shape of affine matrix should be 4x4")

        # Define coil profiles if unshimmed or affine is different from previously
        if (self.unshimmed.shape != unshimmed.shape) or not np.all(self.unshimmed_affine == affine):
            self.merged_coils, self.merged_bounds, self.merged_onoff_channels = self.merge_coils(unshimmed, affine)

        self.merged_bounds_off_channels = [self.merged_bounds[i] for i, is_on in enumerate(self.merged_onoff_channels) if is_on]

        self.unshimmed = unshimmed
        self.unshimmed_affine = affine

    def set_merged_bounds(self, merged_bounds):
        """
        Changes the default bounds set in the coil profile

        Args:
            merged_bounds: Concatenated coil profile bounds. Input all bounds even if some channels are off
        """
        if len(self.merged_bounds) != len(merged_bounds):
            raise ValueError(f"Size of merged bounds: must match the number of total "
                             f"channel: {len(self.merged_bounds)} not {len(merged_bounds)}")

        logger.debug(f"Merged bounds: {merged_bounds}")
        self.merged_bounds = merged_bounds
        self.merged_bounds_off_channels = [self.merged_bounds[i] for i, is_on in enumerate(self.merged_onoff_channels) if is_on]

    def optimize(self, mask, slice_idxs):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (np.ndarray): 3d array marking volume for optimization. Must be the same shape as unshimmed
            slice_idxs (list): List of slice indices being optimized

        Returns:
            np.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """
        coil_mat, unshimmed_vec = self.get_coil_mat_and_unshimmed_masked(mask, slice_idxs)

        # Apply weights to the coil matrix and unshimmed vector
        # The square root of the coefficients is taken since the currents are computed
        # by multiplying two weighted arrays
        coil_mat_w = np.sqrt(self.mask_coefficients)[:, np.newaxis] * coil_mat
        unshimmed_vec_w = np.sqrt(self.mask_coefficients) * unshimmed_vec

        # Scale such that residuals for 0 coefficients gives 1. This is required so that the reg_factor is appropriate
        # for all solves
        factor_fmap = np.linalg.norm(unshimmed_vec_w, ord=2)
        if factor_fmap == 0:
            factor_fmap = 1
        coil_mat_w = coil_mat_w / factor_fmap
        unshimmed_vec_w = unshimmed_vec_w / factor_fmap

        if self.w_signal_loss is not None:
            self._prepare_signal_recovery_data(mask, slice_idxs)
            # Scale such that residuals for 0 coefficients gives 1. This is required so that the reg_factor is appropriate
            # for all solves
            weights_mask = np.sqrt(self.mask_erode_coefficients)
            factor_signal_loss = np.linalg.norm(weights_mask * self.unshimmed_Gz_vec, ord=2)
            if factor_signal_loss == 0:
                factor_signal_loss = 1
            coil_mat_w, unshimmed_vec_w = add_signal_recovery(coil_mat_w,
                                                              unshimmed_vec_w,
                                                              self.w_signal_loss,
                                                              self.coil_Gz_mat * weights_mask[..., np.newaxis] / factor_signal_loss,
                                                              self.unshimmed_Gz_vec * weights_mask / factor_signal_loss)

        if self.reg_factor > 0:
            reg_factor_channel = get_reg_factor_channel(self.merged_bounds_off_channels)
            factor_reg_factor = np.linalg.norm(1 / reg_factor_channel, ord=2)
            if factor_reg_factor == 0:
                factor_reg_factor = 1
            coil_mat_w, unshimmed_vec_w = add_regularization(coil_mat_w, unshimmed_vec_w, self.reg_factor / reg_factor_channel / factor_reg_factor)

        currents = self._get_currents(unshimmed_vec_w, coil_mat_w)

        if logger.level <= getattr(logging, 'DEBUG'):
            # Compute the different obj functions
            # Field
            currents0 = [0,] * len(currents)
            n_values = len(self.mask_coefficients)
            obj_field = np.linalg.norm(coil_mat_w[:n_values, :] @ currents + unshimmed_vec_w[:n_values], ord=2)
            obj_field0 = np.linalg.norm(coil_mat_w[:n_values, :] @ currents0 + unshimmed_vec_w[:n_values], ord=2)
            obj_str = f"RMSE field: {obj_field}"
            obj_str0 = f"RMSE field (before optimization): {obj_field0}"
            start_values = n_values
            if self.w_signal_loss is not None:
                # Signal loss
                n_values = len(self.mask_erode_coefficients)
                obj_sig_loss = np.linalg.norm(coil_mat_w[start_values:start_values+n_values, :] @ currents + unshimmed_vec_w[start_values:start_values+n_values], ord=2)
                obj_sig_loss0 = np.linalg.norm(coil_mat_w[start_values:start_values + n_values, :] @ currents0 + unshimmed_vec_w[start_values:start_values + n_values], ord=2)
                obj_str += f", signal loss: {obj_sig_loss}"
                obj_str0 += f", signal loss (zero): {obj_sig_loss0}"
                start_values += n_values
            if self.reg_factor > 0:
                obj_reg = np.linalg.norm(coil_mat_w[start_values:, :] @ currents, ord=2)
                obj_reg0 = np.linalg.norm(coil_mat_w[start_values:, :] @ currents0, ord=2)
                obj_str += f", regularization factor: {obj_reg}"
                obj_str0 += f", regularization factor (zero): {obj_reg0}"
            logger.debug(obj_str0)
            logger.debug(obj_str)

        currents_all = self.insert_off_channels_values(currents, slice_idxs)

        return currents_all

    def _get_currents(self, unshimmed_vec, coil_mat):
        # Compute the pseudo-inverse of the coil matrix to get the desired coil profiles
        # dimensions : (n_channels, masked_values) @ (masked_values,) --> (n_channels,)
        currents = -1 * scipy.linalg.pinv(coil_mat) @ unshimmed_vec
        return currents

    def get_coil_mat_and_unshimmed_masked(self, mask, slice_idxs):
        """
        Returns the coil matrix, and the unshimmed vector used for the optimization

        Args:
            mask (np.ndarray): 3d array marking volume for optimization. Must be the same shape as unshimmed
            slice_idxs (list): List of slice indices being optimized

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 2D flattened array (masked_values, n_channels) of masked coils
                              (axis 0 must align with unshimmed_vec)
                * np.ndarray: 1D flattened array (masked_values,) of the masked unshimmed map
        """
        # Check for sizing errors
        self._check_sizing(mask)
        # Convert mask to float
        if mask.dtype != float:
            mask = mask.astype(float)
        # Reshape mask to 1D
        mask_vec = mask.reshape((-1,))
        # Get the non-zero mask coefficient values
        self.mask_coefficients = mask_vec[mask_vec != 0]

        merged_coil_opt, unshimmed_opt = self._get_coil_mat_and_unshimmed_on_channels(slice_idxs)

        # Define number of coil profiles (channels)
        n_channels = merged_coil_opt.shape[3] # dimensions : (n_channels,)
        # Transpose coil profile : (X, Y, Z, n_channels) --> (n_channels, X, Y, Z) or (n_channels, [mask.shape])
        merged_coils_transposed = np.transpose(merged_coil_opt, axes=(3, 0, 1, 2))
        # Reshape coil profile : (n_channels, X, Y, Z) --> (n_channels, X * Y * Z) or (n_channels, mask.size)
        merged_coils_reshaped = np.reshape(merged_coils_transposed, (n_channels, -1))

        # Extract the masked coil matrix
        # dimensions : (n_channels, mask.size) --> (mask.size, n_channels) --> (masked_values, n_channels)
        coil_mat = merged_coils_reshaped[:, mask_vec != 0].T
        # Extract the unshimmed vector
        # dimensions : (masked_values,)
        unshimmed_vec = np.reshape(unshimmed_opt, (-1,))[mask_vec != 0]

        return coil_mat, unshimmed_vec

    def _get_coil_mat_and_unshimmed_on_channels(self, slice_idxs):
        """ Returns the coil matrix for the channels that are ON. The values for the channels that are OFF are added
        to the unshimmed vector."""

        # Remove the channels that are off from the merged channels
        if np.sum(self.merged_onoff_channels) < len(self.merged_onoff_channels):
            merged_coil_opt = self.merged_coils[..., self.merged_onoff_channels]
            merged_coil_not_used = self.merged_coils[..., np.logical_not(self.merged_onoff_channels)]
            coefs = self.merge_channels_off_values_single_shim_group(slice_idxs)
            unshimmed_opt = self.unshimmed + merged_coil_not_used @ np.array(coefs)
        else:
            merged_coil_opt = self.merged_coils
            unshimmed_opt = self.unshimmed

        return merged_coil_opt, unshimmed_opt

    def merge_channels_off_values_single_shim_group(self, slice_idxs):
        """ Marge the chim values for the channels that are off for a single shim group for each coil in a single array.

        Args:
            slice_idxs (tuple): Tuple of slice indices being optimized.

        Returns:
            list: List of chim values for the channels that are off for each coil in a single array.
        """
        self._verify_all_channels_off_values_same(slice_idxs)
        coefs = []

        for coil in self.coils:
            if coil.channels_off_values is not None:
                coefs.extend(coil.channels_off_values[slice_idxs[0]])
            elif coil.channels_onoff is not None:
                coefs.extend([0] * np.sum(np.logical_not(coil.channels_onoff)))

        return coefs

    def _verify_all_channels_off_values_same(self, slice_idxs):
        for coil in self.coils:
            if coil.channels_off_values is not None:
                if np.max(slice_idxs) >= coil.channels_off_values.shape[0]:
                    raise ValueError(f"Channels turned OFF fixed values are not the same shape as the number of slices to shim.")
                coil_fixed_values = coil.channels_off_values[slice_idxs[0]]
                for i_slice in range(1, len(slice_idxs)):
                    if not np.allclose(coil.channels_off_values[slice_idxs[i_slice]], coil_fixed_values):
                        raise ValueError(f"Channels_off_values for slice {slice_idxs[i_slice]} are not the same "
                                         f"as slice {slice_idxs[0]}. Values: {coil.channels_off_values[slice_idxs[i_slice]]} vs "
                                         f"{coil_fixed_values}. Since multiple slices are being optimized, the value to use "
                                         f"is ambiguous.")

    def insert_off_channels_values(self, currents, slice_idxs):
        """
        Insert the coefficients for the off channels in the currents array

        Args:
            currents (np.ndarray): 1D array (n_channels_on) containing the coefficients for the channels that are ON
            slice_idxs (tuple): Tuple of slice indices being optimized.

        Returns:
            np.ndarray: 1D array (n_channels) containing the coefficients for all channels, with the OFF channels
                        having their corresponding coefficient from off_channel_coefs
        """
        off_channel_coefs = self.merge_channels_off_values_single_shim_group(slice_idxs)
        new_currents = copy.deepcopy(currents)
        i_off_channel = 0
        for i, is_on in enumerate(self.merged_onoff_channels):
            if not is_on:
                new_currents = np.insert(new_currents, i, off_channel_coefs[i_off_channel])
                i_off_channel += 1

        return new_currents

    def merge_coils(self, unshimmed, affine):
        """
        Uses the list of coil profiles to return a resampled concatenated list of coil profiles matching the
        unshimmed image. Bounds are also concatenated and returned.

        Args:
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
        """

        coil_profiles_list = []

        # Define the nibabel unshimmed array
        nii_unshimmed = nib.Nifti1Image(unshimmed, affine)

        # # Make sure all the coils have the same units
        # units = [coil.units for coil in self.coils]
        # if units.count(units[0]) != len(units):
        #     names = [coil.name for coil in self.coils]
        #     logger.warning(f"The coils don't have matching units: {list(zip(names, units))}")

        for coil in self.coils:
            nii_coil = nib.Nifti1Image(coil.profile, coil.affine)

            # Resample a coil on the unshimmed image
            resampled_coil = resample_from_to(nii_coil, nii_unshimmed).get_fdata()
            coil_profiles_list.append(resampled_coil)

        coil_profiles = np.concatenate(coil_profiles_list, axis=3)

        bounds = self.merge_bounds()
        off_channels = self.merge_off_channels()

        return coil_profiles, bounds, off_channels

    def merge_off_channels(self):
        """
        Merge the coil profile off channels into a single array.

        Returns:
            list: Concatenated list of off channels
        """

        off_channels = []
        for coil in self.coils:
            off_channels.extend(coil.channels_onoff)

        logger.debug(f"Merged off channels: {off_channels}")
        return off_channels

    def merge_bounds(self):
        """
        Merge the coil profile bounds into a single array.

        Returns:
            list: list of bounds corresponding to each merged coils
        """

        bounds = []
        for coil in self.coils:
            # Concat coils and bounds
            for key in coil.coef_channel_minmax:
                for a_bound in coil.coef_channel_minmax[key]:
                    bounds.append(a_bound)

        logger.debug(f"Merged bounds: {bounds}")
        return bounds

    def _check_sizing(self, mask):
        """
        Helper function to check array sizing

        Args:
            mask (np.ndarray): 3d array of integers marking volume for optimization. Must be the same shape as
                                  unshimmed
        """

        if mask.ndim != 3:
            raise ValueError(f"Mask has {mask.ndim} dimensions, expected 3 (dim1, dim2, dim3)")
        if mask.shape != self.unshimmed.shape:
            raise ValueError(f"Mask with shape: {mask.shape} expected to have the same shape as the unshimmed image"
                             f" with shape: {self.unshimmed.shape}")

    def _prepare_signal_recovery_data(self, mask, slice_idxs):
        """ Prepares the data for the optimization.
        """
        # Define coil profiles
        n_channels = np.sum(self.merged_onoff_channels)

        # Remove channels not used in the optimization
        merged_coil_opt, unshimmed_opt = self._get_coil_mat_and_unshimmed_on_channels(slice_idxs)

        # Erode mask
        bin_mask = (mask != 0).astype(int)
        bin_mask_erode = modify_binary_mask(bin_mask, shape='sphere', size=3, operation='erode')
        mask_erode = np.zeros_like(mask, dtype=float)
        mask_erode[bin_mask_erode != 0] = mask[bin_mask_erode != 0]
        mask_erode_vec = mask_erode.reshape((-1,))
        self.mask_erode_coefficients = mask_erode_vec[mask_erode_vec != 0]

        # Define merged coils
        temp = np.transpose(merged_coil_opt, axes=(3, 0, 1, 2))
        merged_coils_Gx = np.zeros(np.shape(temp))
        merged_coils_Gy = np.zeros(np.shape(temp))
        merged_coils_Gz = np.zeros(np.shape(temp))
        for ch in range(n_channels):
            merged_coils_Gx[ch] = np.gradient(temp[ch], axis=0)
            merged_coils_Gy[ch] = np.gradient(temp[ch], axis=1)
            merged_coils_Gz[ch] = np.gradient(temp[ch], axis=2)

        # Define coil matrices for each gradient
        self.coil_Gz_mat = np.reshape(merged_coils_Gz,
                                      (n_channels, -1)).T[mask_erode_vec != 0, :]  # (masked_values, n_channels)
        self.coil_Gx_mat = np.reshape(merged_coils_Gx,
                                      (n_channels, -1)).T[mask_erode_vec != 0, :]  # (masked_values, n_channels)
        self.coil_Gy_mat = np.reshape(merged_coils_Gy,
                                      (n_channels, -1)).T[mask_erode_vec != 0, :]  # (masked_values, n_channels)

        # Define unshimmed vector for each gradient
        self.unshimmed_vec = np.reshape(unshimmed_opt, (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gx_vec = np.reshape(np.gradient(unshimmed_opt, axis=0), (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gy_vec = np.reshape(np.gradient(unshimmed_opt, axis=1), (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gz_vec = np.reshape(np.gradient(unshimmed_opt, axis=2), (-1,))[mask_erode_vec != 0]  # (masked_values,)

        if len(self.unshimmed_Gz_vec) == 0:
            raise ValueError('The mask or the field map is too small to perform the signal recovery optimization. '
                                'Make sure to include at least 3 voxels in the slice direction.')


def get_reg_factor_channel(opt_merged_bounds):
    """
    Sets the regularization factor for each channel based on the bounds of the optimization.

    Args:
        opt_merged_bounds (np.ndarray): 2D array (channel, 2) containing the lower and upper bounds for each
    channel that are ON."""

    reg_factor_channel = np.array([np.abs(bound[1] - bound[0]) for bound in opt_merged_bounds])
    return reg_factor_channel


def add_signal_recovery(coil_mat_w, unshimmed_vec_w, w_signal_loss, coil_G_mat, unshimmed_G_vec):
    # Add signal recovery terms to the optimization
    coil_G_mat_w = coil_G_mat * np.sqrt(w_signal_loss)
    unshimmed_G_vec_w = unshimmed_G_vec * np.sqrt(w_signal_loss)
    return np.vstack([coil_mat_w, coil_G_mat_w]), np.hstack([unshimmed_vec_w, unshimmed_G_vec_w])


def add_regularization(coil_mat_w, unshimmed_vec_w, reg_factor_channel):
    n_channels = coil_mat_w.shape[1]
    l_mat = np.eye(n_channels) * reg_factor_channel
    b_ins = np.zeros(n_channels)
    return np.vstack([coil_mat_w, l_mat]), np.hstack([unshimmed_vec_w, b_ins])
