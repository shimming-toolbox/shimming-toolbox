#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt
from typing import List
import warnings
from shimmingtoolbox.masking.mask_utils import modify_binary_mask

from shimmingtoolbox.optimizer.optimizer_utils import OptimizerUtils
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]
allowed_opt_criteria = ['mse', 'mae', 'std', 'mse_signal_recovery', 'rmse', 'rmse_signal_recovery']
logger = logging.getLogger(__name__)

class LsqOptimizer(OptimizerUtils):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a least squares solver to find the best shim.
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, opt_criteria='mse', initial_guess_method='zeros',
                 reg_factor=0,  w_signal_loss=None, w_signal_loss_xy=None, epi_te=None):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation, 'rmse': root mean squared error.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine, initial_guess_method, reg_factor)
        self.w_signal_loss = w_signal_loss
        self.w_signal_loss_xy = w_signal_loss_xy
        self.epi_te = epi_te
        self.counter = 0
        lsq_residual_dict = {
            allowed_opt_criteria[0]: self._residuals_mse,
            allowed_opt_criteria[1]: self._residuals_mae,
            allowed_opt_criteria[2]: self._residuals_std,
            allowed_opt_criteria[3]: self._residuals_mse_signal_recovery,
            allowed_opt_criteria[4]: self._residuals_rmse,
            allowed_opt_criteria[5]: self._residuals_rmse_signal_recovery
        }
        lsq_jacobian_dict = {
            allowed_opt_criteria[0]: self._residuals_mse_jacobian,
            allowed_opt_criteria[1]: None,
            allowed_opt_criteria[2]: None,
            allowed_opt_criteria[3]: self._residuals_mse_signal_recovery_jacobian,
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
        # Personalized parameters to LSQ
        mask_erode = modify_binary_mask(mask, shape='sphere', size=3, operation='erode')
        mask_erode_vec = mask_erode.reshape((-1,))

        temp = np.transpose(self.merged_coils, axes=(3, 0, 1, 2))
        merged_coils_Gx = np.zeros(np.shape(temp))
        merged_coils_Gy = np.zeros(np.shape(temp))
        merged_coils_Gz = np.zeros(np.shape(temp))
        for ch in range(n_channels):
            merged_coils_Gx[ch] = np.gradient(temp[ch], axis=0)
            merged_coils_Gy[ch] = np.gradient(temp[ch], axis=1)
            merged_coils_Gz[ch] = np.gradient(temp[ch], axis=2)

        self.coil_Gz_mat = np.reshape(merged_coils_Gz,
                                    (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N
        self.coil_Gx_mat = np.reshape(merged_coils_Gx,
                                    (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N
        self.coil_Gy_mat = np.reshape(merged_coils_Gy,
                                    (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N

        self.unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_erode_vec != 0]  # mV'

        self.unshimmed_Gx_vec = np.reshape(np.gradient(self.unshimmed, axis=0), (-1,))[mask_erode_vec != 0]  # mV'
        self.unshimmed_Gy_vec = np.reshape(np.gradient(self.unshimmed, axis=1), (-1,))[mask_erode_vec != 0]  # mV'
        self.unshimmed_Gz_vec = np.reshape(np.gradient(self.unshimmed, axis=2), (-1,))[mask_erode_vec != 0]  # mV'

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

        # MAE regularized to minimize currents
        return np.mean(np.abs(unshimmed_vec + coil_mat @ coef)) / factor + np.abs(coef).dot(self.reg_vector)

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
        shimmed_vec = unshimmed_vec + coil_mat @ coef
        return (shimmed_vec).dot(shimmed_vec) / len(unshimmed_vec) / factor + np.abs(coef).dot(self.reg_vector)

    def _residuals_std(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the standard deviation (STD)

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

        # STD regularized to minimize currents
        return np.std(unshimmed_vec + coil_mat @ coef) / factor + np.abs(coef).dot(self.reg_vector)

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
        b0_rmse_coef = norm((unshimmed_vec + coil_mat @ coef) / factor, 2)
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)

        # RMSE regularized to minimize currents
        return b0_rmse_coef + current_regularization_coef
    
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
        b0_rmse_coef = norm((unshimmed_vec + coil_mat @ coef) / factor, 2)
        signal_recovery_coef = norm((self.unshimmed_Gz_vec + self.coil_Gz_mat @ coef) / factor, 2)
        current_regularization_coef = np.abs(coef).dot(self.reg_vector)
        
        return b0_rmse_coef + signal_recovery_coef * self.w_signal_loss + current_regularization_coef
        
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
            return -1 * (np.sum(np.abs(inputs[indexes])) - coef_sum_max)

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
            # regularization on the currents has no affect on the output stability factor.
            if 'mse' in self.opt_criteria:
                stability_factor = self._initial_guess_mse(self._initial_guess_zeros(), unshimmed_vec,
                                                           np.zeros_like(coil_mat),
                                                           factor=1)
            else:
                stability_factor = self._criteria_func(self._initial_guess_zeros(), unshimmed_vec,
                                                       np.zeros_like(coil_mat),
                                                       factor=1)

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

        # MSE term for unshimmed_vec and coil_mat
        a1 = inv_factor * (coil_mat.T @ coil_mat)
        b1 = 2 * inv_factor * (unshimmed_vec @ coil_mat)
        c1 = inv_factor * (unshimmed_vec @ unshimmed_vec)

        # MSE term for unshimmed_Gz_vec and coil_Gz_mat
        a2 = w_inv_factor_Gz * (self.coil_Gz_mat.T @ self.coil_Gz_mat)
        b2 = 2 * w_inv_factor_Gz * (self.unshimmed_Gz_vec @ self.coil_Gz_mat)
        c2 = w_inv_factor_Gz * (self.unshimmed_Gz_vec @ self.unshimmed_Gz_vec)

        # MSE term for unshimmed_Gx_vec and coil_Gx_mat
        a3 = w_inv_factor_Gxy * (self.coil_Gx_mat.T @ self.coil_Gx_mat)
        b3 = 2 * w_inv_factor_Gxy * (self.unshimmed_Gx_vec @ self.coil_Gx_mat)
        c3 = w_inv_factor_Gxy * (self.unshimmed_Gx_vec @ self.unshimmed_Gx_vec)

        # MSE term for unshimmed_Gy_vec and coil_Gy_mat
        a4 = w_inv_factor_Gxy * (self.coil_Gy_mat.T @ self.coil_Gy_mat)
        b4 = 2 * w_inv_factor_Gxy * (self.unshimmed_Gy_vec @ self.coil_Gy_mat)
        c4 = w_inv_factor_Gxy * (self.unshimmed_Gy_vec @ self.unshimmed_Gy_vec)

        # Combining the terms
        a = a1 + a2 + a3 + a4 + np.diag(self.reg_vector)
        b = b1 + b2 + b3 + b4
        c = c1 + c2 + c3 + c4
        e = self.reg_vector

        return a, b, c, e


class PmuLsqOptimizer(LsqOptimizer):
    """ Optimizer for the realtime component (riro) for this optimization:
        field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
        Unshimmed must be in units: [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

        This optimizer bounds the riro results to the coil bounds by taking the range of pressure that can be reached
        by the PMU.
    """

    def __init__(self, coils, unshimmed, affine, opt_criteria, pmu: PmuResp, reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation, 'rmse': root mean squared error.
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
        """

        super().__init__(coils, unshimmed, affine, opt_criteria, initial_guess_method='zeros', reg_factor=reg_factor)
        self.pressure_min = pmu.min
        self.pressure_max = pmu.max

    def _define_scipy_constraints(self):
        """Redefined from super() to include more constraints"""
        scipy_constraints = self._define_scipy_coef_sum_max_constraint()
        scipy_constraints += self._define_scipy_bounds_constraint()
        return scipy_constraints

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

    def _define_scipy_bounds_constraint(self):
        """Constraints that allows to respect the bounds. Since the pressure can vary up and down, there are 2 maximum
        values that the currents can have for each optimization. This allows to set a constraint that multiplies the
        maximum pressure value by the currents of units [unit_pressure / unit_currents] and see if it respects the
        bounds then do the same thing for the mimimum pressure value.
        """

        def _apply_min_pressure_constraint(inputs, i_channel, bound_min, pressure_min):
            # ineq constraint for scipy minimize function. Negative output is disregarded while positive output is kept.
            # Make sure the min current for a channel is higher than the min bound for that channel
            current_min_pressure = inputs[i_channel] * pressure_min

            # current_min_pressure > bound_min
            return current_min_pressure - bound_min

        def _apply_max_pressure_constraint(inputs, i_channel, bound_max, pressure_max):
            # ineq constraint for scipy minimize function. Negative output is disregarded while positive output is kept.
            # Make sure the max current for a channel is lower than the max bound for that channel
            current_max_pressure = inputs[i_channel] * pressure_max

            # current_max_pressure < bound_max
            return bound_max - current_max_pressure

        # riro_offset = riro * (acq_pressure - mean_p)
        # Maximum/minimum values of riro_offset will occure when (acq_pressure - mean_p) are min and max:
        # if the mean is self.pressure_min and the pressure probe reads self.pressure_max --> delta_pressure
        # if the mean is self.pressure_max and the pressure probe reads self.pressure_min --> -delta_pressure
        # Those are theoretical min and max, they should not happen since the mean should preferably be in the middle of
        # the probe's min/max
        delta_pressure = self.pressure_max - self.pressure_min

        constraints = []
        for i_bound, bound in enumerate(self.merged_bounds):
            # No point in adding a constraint if the bound is infinite
            if not bound[0] == -np.inf:
                # Minimum bound
                constraints.append({'type': 'ineq', "fun": _apply_min_pressure_constraint,
                                    'args': (i_bound, bound[0], -delta_pressure)})
                constraints.append({'type': 'ineq', "fun": _apply_min_pressure_constraint,
                                    'args': (i_bound, bound[0], delta_pressure)})
            # No point in adding a constraint if the bound is infinite
            if not bound[1] == np.inf:
                # Maximum bound
                constraints.append({'type': 'ineq', "fun": _apply_max_pressure_constraint,
                                    'args': (i_bound, bound[1], delta_pressure)})
                constraints.append({'type': 'ineq', "fun": _apply_max_pressure_constraint,
                                    'args': (i_bound, bound[1], -delta_pressure)})

        return constraints

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        """Redefined from super() since normal bounds are now constraints"""
        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)

            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='SLSQP',
                                       constraints=tuple(scipy_constraints),
                                       jac=self._jacobian_func,
                                       options={'maxiter': 1000, 'ftol': 1e-9})

        else:
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(unshimmed_vec, coil_mat, factor),
                                       method='SLSQP',
                                       constraints=tuple(scipy_constraints),
                                       jac=self._jacobian_func,
                                       options={'maxiter': 1000, 'ftol': 1e-9})

        return currents_sp
