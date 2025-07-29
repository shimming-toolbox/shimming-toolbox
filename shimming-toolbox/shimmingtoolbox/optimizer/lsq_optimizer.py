#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt
from scipy.special import pseudo_huber
from typing import List
import warnings
from shimmingtoolbox.masking.mask_utils import modify_binary_mask
from scipy.ndimage import map_coordinates

from shimmingtoolbox.optimizer.optimizer_utils import OptimizerUtils
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]
allowed_opt_criteria = ['mse', 'mae', 'mse_signal_recovery', 'rmse', 'rmse_signal_recovery', 'ps_huber']
logger = logging.getLogger(__name__)

class LsqOptimizer(OptimizerUtils):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a least squares solver to find the best shim.
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, opt_criteria='mse', initial_guess_method='zeros',
                 reg_factor=0, w_signal_loss=None, w_signal_loss_xy=None, epi_te=None):
        f"""
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. {allowed_opt_criteria}.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine, initial_guess_method, reg_factor)

        self._delta = None  # Initialize delta for pseudo huber function

        # Initialization of grad parameters
        self.w_signal_loss = w_signal_loss
        self.w_signal_loss_xy = w_signal_loss_xy
        self.epi_te = epi_te
        self.counter = 0

        lsq_residual_dict = {
            allowed_opt_criteria[0]: self._residuals_mse,
            allowed_opt_criteria[1]: self._residuals_mae,
            allowed_opt_criteria[2]: self._residuals_mse_signal_recovery,
            allowed_opt_criteria[3]: self._residuals_rmse,
            allowed_opt_criteria[4]: self._residuals_rmse_signal_recovery,
            allowed_opt_criteria[5]: self._residuals_ps_huber
        }
        lsq_jacobian_dict = {
            allowed_opt_criteria[0]: self._residuals_mse_jacobian,
            allowed_opt_criteria[1]: None,
            allowed_opt_criteria[2]: self._residuals_mse_signal_recovery_jacobian,
            allowed_opt_criteria[3]: None,
            allowed_opt_criteria[4]: None,
            allowed_opt_criteria[5]: None
        }

        if opt_criteria in allowed_opt_criteria:
            self._criteria_func = lsq_residual_dict[opt_criteria]
            self._jacobian_func = lsq_jacobian_dict[opt_criteria]
            self.opt_criteria = opt_criteria
        else:
            raise ValueError("Optimization criteria not supported")

    def _prepare_signal_recovery_data(self, mask):
        """ Prepares the data for the optimization.
        """
        self.counter += 1
        # Define coil profiles
        n_channels = self.merged_coils.shape[3]
        # Erode mask
        bin_mask = (mask != 0).astype(int)
        bin_mask_erode = modify_binary_mask(bin_mask, shape='sphere', size=3, operation='erode')
        mask_erode = np.zeros_like(mask, dtype=float)
        mask_erode[bin_mask_erode != 0] = mask[bin_mask_erode != 0]
        mask_erode_vec = mask_erode.reshape((-1,))

        # Define merged coils
        temp = np.transpose(self.merged_coils, axes=(3, 0, 1, 2))
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
        self.unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gx_vec = np.reshape(np.gradient(self.unshimmed, axis=0), (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gy_vec = np.reshape(np.gradient(self.unshimmed, axis=1), (-1,))[mask_erode_vec != 0]  # (masked_values,)
        self.unshimmed_Gz_vec = np.reshape(np.gradient(self.unshimmed, axis=2), (-1,))[mask_erode_vec != 0]  # (masked_values,)

        # Interpolated mask coefficients at gradient centers (using map_coordinates)
        shape_x, shape_y, shape_z = mask_erode.shape

        # Gx: centers between x-voxels, so x in [0.5, ..., shape_x-1.5], y in [0, ..., shape_y-1], z in [0, ..., shape_z-1]
        gx_x = np.arange(shape_x - 1) + 0.5
        gx_y = np.arange(shape_y)
        gx_z = np.arange(shape_z)
        gx_coords = np.meshgrid(gx_x, gx_y, gx_z, indexing='ij')
        coords_Gx = np.vstack([c.ravel() for c in gx_coords])
        mask_Gx_interp = map_coordinates(mask_erode, coords_Gx, order=1, mode='nearest')

        # Gy: centers between y-voxels, so y in [0.5, ..., shape_y-1.5]
        gy_x = np.arange(shape_x)
        gy_y = np.arange(shape_y - 1) + 0.5
        gy_z = np.arange(shape_z)
        gy_coords = np.meshgrid(gy_x, gy_y, gy_z, indexing='ij')
        coords_Gy = np.vstack([c.ravel() for c in gy_coords])
        mask_Gy_interp = map_coordinates(mask_erode, coords_Gy, order=1, mode='nearest')

        # Gz: centers between z-voxels, so z in [0.5, ..., shape_z-1.5]
        gz_x = np.arange(shape_x)
        gz_y = np.arange(shape_y)
        gz_z = np.arange(shape_z - 1) + 0.5
        gz_coords = np.meshgrid(gz_x, gz_y, gz_z, indexing='ij')
        coords_Gz = np.vstack([c.ravel() for c in gz_coords])
        mask_Gz_interp = map_coordinates(mask_erode, coords_Gz, order=1, mode='nearest')

        # Select the coefficients corresponding to the eroded mask.
        # The coil matrices (e.g., coil_Gx_mat) are masked using mask_erode_vec != 0, so we need to apply
        # a similar masking to the interpolated mask coefficients.
        # However, the interpolated masks (e.g., mask_Gx_interp) have smaller dimensions due to the gradient computation
        # (e.g., shape_x - 1 for Gx), so we must crop the eroded mask accordingly before flattening.
        # This ensures that the mask used for indexing has the same shape as the interpolated values.
        mask_erode_Gx = mask_erode[:-1, :, :]
        mask_erode_Gy = mask_erode[:, :-1, :]
        mask_erode_Gz = mask_erode[:, :, :-1]
        mask_erode_vec_Gx = mask_erode_Gx.reshape(-1)
        mask_erode_vec_Gy = mask_erode_Gy.reshape(-1)
        mask_erode_vec_Gz = mask_erode_Gz.reshape(-1)

        self.mask_Gx_coefficients = mask_Gx_interp[mask_erode_vec_Gx != 0]
        self.mask_Gy_coefficients = mask_Gy_interp[mask_erode_vec_Gy != 0]
        self.mask_Gz_coefficients = mask_Gz_interp[mask_erode_vec_Gz != 0]

        if len(self.unshimmed_Gz_vec) == 0:
            raise ValueError('The mask or the field map is too small to perform the signal recovery optimization. '
                                'Make sure to include at least 3 voxels in the slice direction.')

    def optimize(self, mask):
        """
        Wrapper for the optimization function. This function prepares the data and calls the optimizer.
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (np.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).

        Returns:
            np.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                            The shape of the array returned has shape corresponding to the total number of channels
        """
        if 'signal_recovery' in self.opt_criteria:
            self._prepare_signal_recovery_data(mask)

        self.mask = mask
        coil_mat, unshimmed_vec = self.get_coil_mat_and_unshimmed(mask)
        # Set up output currents
        currents_0 = self.get_initial_guess()
        # If what to shim is already 0s
        if np.all(unshimmed_vec == 0):
            return np.zeros(np.shape(currents_0))
        currents = self._get_currents(unshimmed_vec, coil_mat, currents_0)

        return currents

    def _residuals_mae(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the mean absolute error (MAE)

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization
        """
        # Apply weights to the coil matrix and unshimmed vector
        weights = self.mask_coefficients
        coil_mat_w = weights[:, np.newaxis] * coil_mat
        unshimmed_vec_w = weights * unshimmed_vec

        # Compute the MAE residuals
        shimmed_vec_w = unshimmed_vec_w + coil_mat_w @ coef
        mae = np.sum(np.abs(shimmed_vec_w)) / np.sum(weights)
        mae_coef = mae / factor # MAE regularized to minimize currents
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        return mae_coef + current_regularization_coef

    def _residuals_ps_huber(self, coef, unshimmed_vec, coil_mat, factor):
        """ Pseudo huber objective function to minimize mean squared error or mean absolute error.
        The delta parameter defines a threshold between the linear (mae) vs. quadratic (mse) behavior and
        is calculated from the residuals of the field.

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization
        """
        # Define the weights and the shimmed vector
        weights = self.mask_coefficients
        shimmed_vec = unshimmed_vec + coil_mat @ coef

        # Define delta with the 90th percentile of the absolute shimmed vector
        if self._delta is None:
            self._delta = np.percentile(np.abs(shimmed_vec), 90)

        # Compute the mean pseudo huber (MPSH) residuals
        mpsh = np.sum(weights * pseudo_huber(self._delta, shimmed_vec)) / np.sum(weights)
        mpsh_coeff = mpsh / factor # Pseudo huber regularized to minimize currents
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        return mpsh_coeff + current_regularization_coef

    def _residuals_mse(self, coef, a, b, c):
        """ Objective function to minimize the mean squared error (MSE)

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            a (np.ndarray): 2D array used for the optimization
            b (np.ndarray): 1D flattened array used for the optimization
            c (float) : Float used for the optimization

        Returns:
            float: Residuals for least squares optimization
        """
        # The first version was :
        # np.mean((unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False))**2) / factor + \ (
        # self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))
        # For the second version we switched np.sum(coil_mat*coef,axis=1,keepdims=False) by coil_mat@coef
        # which is way faster Then for a vector, mean(x**2) is equivalent to x.dot( x)/n
        # it's faster to do this operation instead of a np.mean Finally np.abs(coef).dot(self.reg_vector) is
        # equivalent and faster to self.reg_factor*np.mean(np.abs(coef) / self.reg_factor_channel) For the
        # mathematical demonstration see : https://github.com/shimming-toolbox/shimming-toolbox/pull/432
        # This gave us the following expression for the residuals mse :
        # shimmed_vec = unshimmed_vec + coil_mat @ coef
        # mse = (shimmed_vec).dot(shimmed_vec) / len(unshimmed_vec) / factor + np.abs(coef).dot(self.reg_vector)
        # The new expression of residuals mse, is the fastest way to the optimization because it allows us to not
        # calculate everytime some long processing operation, the term of a, b and c were calculated in scipy_minimize
        return a @ coef @ coef + b @ coef + c

    def _residuals_mse_signal_recovery(self, coef, a, b, c, e):
        result = coef.T @ a @ coef + b @ coef + c + np.abs(coef) @ e
        return result

    def _initial_guess_mse(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to find the initial guess for the mean squared error (MSE) optimization

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization
        """
        # Apply weights to the coil matrix and unshimmed vector
        weights = np.sqrt(self.mask_coefficients)
        coil_mat_w = weights[:, np.newaxis] * coil_mat
        unshimmed_vec_w = weights * unshimmed_vec

        # Compute the MSE residuals
        shimmed_vec_w = unshimmed_vec_w + coil_mat_w @ coef
        mse = np.sum(np.square(shimmed_vec_w)) / np.sum(np.square(weights))
        mse_coef = mse / factor  # MSE regularized to minimize currents
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        return mse_coef + current_regularization_coef

    def _residuals_rmse(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the root mean squared error (RMSE)

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization
        """
        # Apply weights to the coil matrix and unshimmed vector
        weights = np.sqrt(self.mask_coefficients)
        coil_mat_w = weights[:, np.newaxis] * coil_mat
        unshimmed_vec_w = weights * unshimmed_vec

        # Compute the RMSE residuals
        shimmed_vec_w = unshimmed_vec_w + coil_mat_w @ coef
        mse = np.sum(np.square(shimmed_vec_w)) / np.sum(np.square(weights))
        rmse_coef = np.sqrt(mse) / factor  # RMSE regularized to minimize currents
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        return rmse_coef + current_regularization_coef

    def _residuals_rmse_signal_recovery(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the root mean squared error (RMSE)
            with through-slice gradient minimization

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization with through-slice gradient minimization
        """
        # Apply weights to the coil matrix and unshimmed vector (B0)
        weights = np.sqrt(self.mask_coefficients)
        coil_mat_w = weights[:, np.newaxis] * coil_mat
        unshimmed_vec_w = weights * unshimmed_vec
        # Apply weights to the coil matrix and unshimmed vector (Gz)
        weights_Gz = np.sqrt(self.mask_Gz_coefficients)
        coil_Gz_mat_w = weights_Gz[:, np.newaxis] * self.coil_Gz_mat
        unshimmed_Gz_vec_w = weights_Gz * self.unshimmed_Gz_vec

        # Compute the RMSE residuals (B0)
        shimmed_vec_w = unshimmed_vec_w + coil_mat_w @ coef
        mse_b0 = np.sum(np.square(shimmed_vec_w)) / np.sum(np.square(weights))
        rmse_b0_coef = np.sqrt(mse_b0) / factor # RMSE regularized to minimize currents
        # Compute the RMSE residuals (Gz)
        shimmed_vec_w_Gz = unshimmed_Gz_vec_w + coil_Gz_mat_w @ coef
        mse_Gz = np.sum(np.square(shimmed_vec_w_Gz)) / np.sum(np.square(weights_Gz))
        rmse_Gz_coef = np.sqrt(mse_Gz) / factor # RMSE regularized to minimize currents
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        return rmse_b0_coef + rmse_Gz_coef * self.w_signal_loss + current_regularization_coef

    def _residuals_mse_jacobian(self, coef, a, b, c):
        """ Jacobian of the function that we want to minimize
        The function to minimize is :
        np.mean((unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)) ** 2) / factor+\
           (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))
        The first Version of the jacobian was :
        self.b * (unshimmed_vec + coil_mat @ coef) @ coil_mat + np.sign(coef) * self.reg_vector where self.b was equal
        to 2/(len(unshimmed_vec * factor)
        This jacobian come from the new version of the residuals mse that was implemented with the PR#451

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            a (np.ndarray): 2D array using for the optimization
            b (np.ndarray): 1D flattened array used for the optimization
            c (float) : Float used for the optimization but not used here

        Returns:
            np.ndarray : 1D array of the gradient of the mse function to minimize
        """
        return 2 * a @ coef + b

    def _residuals_mse_signal_recovery_jacobian(self, coef, a, b, c, e):
        """ Jacobian of the function that we want to minimize

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            a (np.ndarray): 2D array using for the optimization
            b (np.ndarray): 1D flattened array used for the optimization
            c (float) : Float used for the optimization but not used here
            e (np.ndarray): 1D array of the regularization vector

        Returns:
            np.ndarray : 1D array of the gradient of the mse function to minimize
        """
        return 2 * a @ coef + b + np.sign(coef) * e

    def _define_scipy_constraints(self):
        return self._define_scipy_coef_sum_max_constraint()

    def _define_scipy_coef_sum_max_constraint(self):
        """Constraint on each coil about the maximum current of all channels"""

        def _apply_sum_constraint(inputs, indexes, coef_sum_max):
            # ineq constraint for scipy minimize function. Negative output is disregarded while positive output is kept.
            return -np.sum(np.abs(inputs[indexes])) + coef_sum_max

        # Set up constraints for max current for each coils
        constraints = []
        start_index = 0
        for i_coil in range(len(self.coils)):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                constraints.append({'type': 'ineq', "fun": _apply_sum_constraint,
                                    'args': (range(start_index, end_index), coil.coef_sum_max)})
            start_index = end_index
        return constraints

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='SLSQP',
                                       bounds=self.merged_bounds,
                                       constraints=tuple(scipy_constraints),
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        elif self.opt_criteria == 'mse_signal_recovery':
            a, b, c, e = self.get_quadratic_term_grad(unshimmed_vec, coil_mat, factor)

            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c, e),
                                       method='SLSQP',
                                       bounds=self.merged_bounds,
                                       constraints=tuple(scipy_constraints),
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        else:
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(unshimmed_vec, coil_mat, factor),
                                       method='SLSQP',
                                       bounds=self.merged_bounds,
                                       constraints=tuple(scipy_constraints),
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        return currents_sp

    def _get_currents(self, unshimmed_vec, coil_mat, currents_0):

        scipy_constraints = self._define_scipy_constraints()
        # Optimize
        # When clipping to bounds, scipy raises a warning. Since this can be frequent for our purposes, we ignore that
        # warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="Values in x were outside bounds during a "
                                                      "minimize step, clipping to bounds",
                                    category=RuntimeWarning,
                                    module='scipy')
            # scipy minimize expects the return value of the residual function to be ~10^0 to 10^1
            # --> aiming for 1 then optimizing will lower that. We are using an initial guess of 0s so that the
            # regularization on the currents has no effect on the output stability factor.
            if self.opt_criteria in ['mse', 'mse_signal_recovery']:
                stability_factor = self._initial_guess_mse(self._initial_guess_zeros(), unshimmed_vec,
                                                           np.zeros_like(coil_mat),
                                                           factor=1)
            else:
                stability_factor = self._criteria_func(self._initial_guess_zeros(), unshimmed_vec,
                                                       np.zeros_like(coil_mat),
                                                       factor=1)
            # Reset delta for pseudo huber function
            self._delta = None
            currents_sp = self._scipy_minimize(currents_0, unshimmed_vec, coil_mat, scipy_constraints,
                                               factor=stability_factor)
        if not currents_sp.success:
            raise RuntimeError(f"Optimization failed due to: {currents_sp.message}")

        currents = currents_sp.x

        return currents

    def get_quadratic_term_grad(self, unshimmed_vec, coil_mat, factor):
        len_unshimmed = len(unshimmed_vec)
        len_unshimmed_Gz = len(self.unshimmed_Gz_vec)
        len_unshimmed_Gx = len(self.unshimmed_Gx_vec)

        inv_factor = 1 / (len_unshimmed * factor)
        w_inv_factor_Gz = self.w_signal_loss / len_unshimmed_Gz
        w_inv_factor_Gxy = self.w_signal_loss_xy / len_unshimmed_Gx

        # Apply weights to the coil matrices and unshimmed vectors
        coil_mat_w = self.mask_coefficients[:, np.newaxis] * coil_mat
        unshimmed_vec_w = self.mask_coefficients * unshimmed_vec
        coil_Gz_mat_w = self.mask_Gz_coefficients[:, np.newaxis] * self.coil_Gz_mat
        unshimmed_Gz_vec_w = self.mask_Gz_coefficients * self.unshimmed_Gz_vec
        coil_Gx_mat_w = self.mask_Gx_coefficients[:, np.newaxis] * self.coil_Gx_mat
        unshimmed_Gx_vec_w = self.mask_Gx_coefficients * self.unshimmed_Gx_vec
        coil_Gy_mat_w = self.mask_Gy_coefficients[:, np.newaxis] * self.coil_Gy_mat
        unshimmed_Gy_vec_w = self.mask_Gy_coefficients * self.unshimmed_Gy_vec

        # MSE term for unshimmed_vec and coil_mat
        a1 = inv_factor * (coil_mat_w.T @ coil_mat_w)
        b1 = 2 * inv_factor * (unshimmed_vec_w @ coil_mat_w)
        c1 = inv_factor * (unshimmed_vec_w @ unshimmed_vec_w)

        # MSE term for unshimmed_Gz_vec and coil_Gz_mat
        a2 = w_inv_factor_Gz * (coil_Gz_mat_w.T @ coil_Gz_mat_w)
        b2 = 2 * w_inv_factor_Gz * (unshimmed_Gz_vec_w @ coil_Gz_mat_w)
        c2 = w_inv_factor_Gz * (unshimmed_Gz_vec_w @ unshimmed_Gz_vec_w)

        # MSE term for unshimmed_Gx_vec and coil_Gx_mat
        a3 = w_inv_factor_Gxy * (coil_Gx_mat_w.T @ coil_Gx_mat_w)
        b3 = 2 * w_inv_factor_Gxy * (unshimmed_Gx_vec_w @ coil_Gx_mat_w)
        c3 = w_inv_factor_Gxy * (unshimmed_Gx_vec_w @ unshimmed_Gx_vec_w)

        # MSE term for unshimmed_Gy_vec and coil_Gy_mat
        a4 = w_inv_factor_Gxy * (coil_Gy_mat_w.T @ coil_Gy_mat_w)
        b4 = 2 * w_inv_factor_Gxy * (unshimmed_Gy_vec_w @ coil_Gy_mat_w)
        c4 = w_inv_factor_Gxy * (unshimmed_Gy_vec_w @ unshimmed_Gy_vec_w)

        # Combining the terms
        a = a1 + a2 + a3 + a4 + np.diag(self.reg_vector)
        b = b1 + b2 + b3 + b4
        c = c1 + c2 + c3 + c4
        e = self.reg_vector

        return a, b, c, e

# TODO : Realtime softmask B0 shimming needs to be implemented
class PmuLsqOptimizer(LsqOptimizer):
    """ Optimizer for the realtime component (riro) for this optimization:
        field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
        Unshimmed must be in units: [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

        This optimizer bounds the riro results to the coil bounds by taking the range of pressure that can be reached
        by the PMU.
    """

    def __init__(self, coils, unshimmed, affine, opt_criteria, pmu: PmuResp, mean_p=0, reg_factor=0):
        f"""
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. {allowed_opt_criteria}.
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
            mean_p (float): Mean pressure value during the acquisition.
        """

        super().__init__(coils, unshimmed, affine, opt_criteria, initial_guess_method='zeros', reg_factor=reg_factor)
        self.pressure_min = pmu.min
        self.pressure_max = pmu.max
        self.pressure_mean = mean_p
        self.rt_bounds = None

    def _define_scipy_constraints(self):
        """Redefined from super()"""
        scipy_constraints = self._define_scipy_coef_sum_max_constraint()
        self.rt_bounds = self.define_rt_bounds()
        return scipy_constraints

    def define_rt_bounds(self):
        """ Define bounds taking into account that the formula scales the coefficient by the acquired pressure.

        riro_offset = riro * (acq_pressure - mean_p)

        Since the pressure can vary up and down, there are 2 maximum and 2 minimum values that the currents can have.
        We select the lower and greater of the 2 values respectively.
        """
        # coef * (self.max_pressure - self.pressure_mean) < bound_max
        # coef * (self.min_pressure - self.pressure_mean) < bound_max
        # coef * (self.max_pressure - self.pressure_mean) > bound_min
        # coef * (self.min_pressure - self.pressure_mean) > bound_min
        # =>
        # coef  < bound_max / (self.max_pressure - self.pressure_mean)
        # coef  > bound_max / (self.min_pressure - self.pressure_mean)
        # coef  > bound_min / (self.max_pressure - self.pressure_mean)
        # coef  < bound_min / (self.min_pressure - self.pressure_mean)
        rt_bounds = []
        for i_bound, bound in enumerate(self.merged_bounds):
            tmp_bound = []

            # print(bound[0] / (self.pressure_max - self.pressure_mean))  # Must be greater than this value
            # print(bound[0] / (self.pressure_min - self.pressure_mean))  # Must be lower than this value
            # print(bound[1] / (self.pressure_max - self.pressure_mean))  # Must be lower than this value
            # print(bound[1] / (self.pressure_min - self.pressure_mean))  # Must be greater than this value

            # No point in adding a constraint if the bound is infinite
            if not bound[0] == -np.inf:
                tmp_bound.append(max(bound[0] / (self.pressure_max - self.pressure_mean),
                                     bound[1] / (self.pressure_min - self.pressure_mean)))
            else:
                tmp_bound.append(-np.inf)
            # No point in adding a constraint if the bound is infinite
            if not bound[1] == np.inf:
                tmp_bound.append(min(bound[0] / (self.pressure_min - self.pressure_mean),
                                     bound[1] / (self.pressure_max - self.pressure_mean)))
            else:
                tmp_bound.append(np.inf)

            if tmp_bound[0] > tmp_bound[1]:
                raise ValueError("The bounds are not consistent with the pressure range.")
            rt_bounds.append(tmp_bound)
        return rt_bounds

    def _define_scipy_coef_sum_max_constraint(self):
        """Constraint on each coil about the maximum current of all channels"""

        def _apply_sum_min_pressure_constraint(inputs, indexes, coef_sum_max, min_pressure):
            # ineq constraint for scipy minimize function. Negative output is disregarded while positive output is kept.
            currents_min_pressure = inputs * min_pressure

            # sum_at_min_pressure < coef_sum_max
            return coef_sum_max - np.sum(np.abs(currents_min_pressure[indexes]))

        def _apply_sum_max_pressure_constraint(inputs, indexes, coef_sum_max, max_pressure):
            # ineq constraint for scipy minimize function. Negative output is disregarded while positive output is kept.
            currents_max_pressure = inputs * max_pressure

            # sum_at_max_pressure < coef_sum_max
            return coef_sum_max - np.sum(np.abs(currents_max_pressure[indexes]))

        # Set up constraints for max current for each coils
        constraints = []
        start_index = 0
        for i_coil in range(len(self.coils)):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                constraints.append({'type': 'ineq', "fun": _apply_sum_min_pressure_constraint,
                                    'args': (range(start_index, end_index), coil.coef_sum_max, self.pressure_min)})
                constraints.append({'type': 'ineq', "fun": _apply_sum_max_pressure_constraint,
                                    'args': (range(start_index, end_index), coil.coef_sum_max, self.pressure_max)})
            start_index = end_index
        return constraints

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        """Redefined from super() since normal bounds are now constraints"""
        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)

            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='SLSQP',
                                       bounds=self.rt_bounds,
                                       constraints=tuple(scipy_constraints),
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})
        else:
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(unshimmed_vec, coil_mat, factor),
                                       method='SLSQP',
                                       bounds=self.rt_bounds,
                                       constraints=tuple(scipy_constraints),
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        return currents_sp
