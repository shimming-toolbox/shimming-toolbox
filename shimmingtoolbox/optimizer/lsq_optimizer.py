#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
from typing import List
import warnings
from shimmingtoolbox.masking.mask_utils import erode_binary_mask

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]
# add allowed criteria 'grad'
allowed_opt_criteria = ['mse', 'mae', 'std', 'grad']


class LsqOptimizer(Optimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a least squares solver to find the best shim.allowed_opt_criteria
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, opt_criteria='mse', reg_factor=0, w_signal_loss=0, epi_te=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine)
        self._initial_guess_method = 'mean'
        self.initial_coefs = None
        self.reg_factor = reg_factor
        self.w_signal_loss = w_signal_loss
        self.epi_te = epi_te
        self.reg_factor_channel = np.array([max(np.abs(bound)) for bound in self.merged_bounds])

        lsq_residual_dict = {
            allowed_opt_criteria[0]: self._residuals_mse,
            allowed_opt_criteria[1]: self._residuals_mae,
            allowed_opt_criteria[2]: self._residuals_std,
            #############################################
            ####### Yixin add the following code ########
            allowed_opt_criteria[3]: self._residuals_grad
            #############################################
        }
        if opt_criteria in lsq_residual_dict:
            self._criteria_func = lsq_residual_dict[opt_criteria]
        else:
            raise ValueError("Optimization criteria not supported")

    @property
    def initial_guess_method(self):
        return self._initial_guess_method

    @initial_guess_method.setter
    def initial_guess_method(self, method, coefs=None):
        allowed_methods = ['mean', 'zeros', 'set']
        if method not in allowed_methods:
            raise ValueError(f"Initial_guess_method not supported. Supported methods are: {allowed_methods}")

        if method == 'set':
            if coefs is not None:
                self.initial_coefs = coefs
            else:
                raise ValueError(f"There are no coefficients to set")

        self._initial_guess_method = method

    def _residuals_mae(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the mean absolute error (MAE)

        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector
        """

        # MAE regularized to minimize currents
        return np.mean(np.abs(unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False))) / factor + \
            (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))
            
    def _residuals_grad_orig(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the mean squared error (MSE) and the signal loss function (gradient in z direction)

        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch
        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector
        """
        #print("w_signal_loss is: " + str(self.w_signal_loss) + "," + " epi_te is: " + str(self.epi_te) + " factor is" + str(factor))
        nx,ny,nz,nc = np.shape(self.merged_coils)
        shimmed = self.unshimmed + np.sum(self.merged_coils * np.tile(coef,(nx,ny,nz,1)),axis= 3) # need test
        signal = 1
        # if consider signal loss from x, y, and z
        for i in range(2,3):
            G = np.gradient(shimmed, axis = i)
            signal = signal * abs(np.sinc(self.epi_te * G))
        # MSE regularized to minimize currents
        #print("" + str(np.shape(signal)))
        #print("in this round of optimization, residual from signal loss is : " + str(np.mean(1 - signal) * self.w_signal_loss) + ", residual from B0 inhomogeneity is: " + str(np.mean(np.abs(unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)))) + ", residual from current is " + str((self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))))
        return np.mean(1 - signal) + \
               np.mean((unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)) ** 2) / factor * self.w_signal_loss + \
               (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))
    
    def _residuals_grad(self, coef, unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, factor):
        #print("current Gradient residual is:" + str(np.mean((unshimmed_Gz_vec + np.sum(coil_Gz_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss + \
        #               np.mean((unshimmed_Gx_vec + np.sum(coil_Gx_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss + \
        #               np.mean((unshimmed_Gy_vec + np.sum(coil_Gy_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss))
        return np.mean((unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)) ** 2) / factor + \
               (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel)) + \
               np.mean((unshimmed_Gz_vec + np.sum(coil_Gz_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss
            #  np.mean((unshimmed_Gx_vec + np.sum(coil_Gx_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss + \
            #  np.mean((unshimmed_Gy_vec + np.sum(coil_Gy_mat * coef, axis=1, keepdims=False)) ** 2) * self.w_signal_loss + \
               
        
    def _residuals_mse(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the mean squared error (MSE)

        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector
        """

        # MSE regularized to minimize currents
        return np.mean((unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)) ** 2) / factor + \
            (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))

    def _residuals_std(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to minimize the standard deviation (STD)

        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector
        """

        # STD regularized to minimize currents
        return np.std(unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)) / factor + \
            (self.reg_factor * np.mean(np.abs(coef) / self.reg_factor_channel))

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

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, scipy_constraints, factor):

        currents_sp = opt.minimize(self._criteria_func, currents_0,
                                   args=(unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, factor),
                                   method='SLSQP',
                                   bounds=self.merged_bounds,
                                   constraints=tuple(scipy_constraints),
                                   options={'maxiter': 1000})
        return currents_sp

    def get_initial_guess(self):
        """ Calculates the initial guess according to the `self.initial_guess_method`

        Returns:
            np.ndarray: 1d array (n_channels) containing the initial guess for the optimization
        """

        allowed_guess_method = {
            'mean': self._initial_guess_mean_bounds,
            'zeros': self._initial_guess_zeros,
            'set': self._initial_guess_set
        }

        initial_guess = allowed_guess_method[self.initial_guess_method]()

        return initial_guess

    def _initial_guess_set(self):
        return self.initial_coefs

    def _initial_guess_mean_bounds(self):
        """
        Calculates the initial guess from the bounds, sets it to the mean of the bounds

        Returns:
            np.ndarray: 1d array (n_channels) of coefficient representing the initial guess

        """
        current_0 = []
        for bounds in self.merged_bounds:
            avg = np.mean(bounds)

            if np.isnan(avg):
                current_0.append(0)
            else:
                current_0.append(avg)

        return np.array(current_0)

    def _initial_guess_zeros(self):
        """
        Return a numpy array with zeros.

        Returns:
            np.ndarray: 1d array (n_channels) of coefficient representing the initial guess

        """
        current_0 = np.zeros(len(self.merged_bounds))

        return current_0

    def optimize(self, mask):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            mask (numpy.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).

        Returns:
            numpy.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """

        # Check for sizing errors
        self._check_sizing(mask)

        # Define coil profiles
        n_channels = self.merged_coils.shape[3]
        mask_vec = mask.reshape((-1,))
        
        mask_erode = erode_binary_mask(mask,shape='sphere',size = 3)
        mask_erode_vec = mask.reshape((-1,))
        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        temp = np.transpose(self.merged_coils, axes=(3, 0, 1, 2))
        coil_mat = np.reshape(temp,
                              (n_channels, -1)).T[mask_vec != 0, :]  # masked points x N
        
        merged_coils_Gx = np.zeros(np.shape(temp))
        merged_coils_Gy = np.zeros(np.shape(temp))
        merged_coils_Gz = np.zeros(np.shape(temp))
        for ch in range(n_channels):
            merged_coils_Gx[ch] = np.gradient(temp[ch], axis = 0)
            merged_coils_Gy[ch] = np.gradient(temp[ch], axis = 1)
            merged_coils_Gz[ch] = np.gradient(temp[ch], axis = 2)
            
        coil_Gx_mat = np.reshape(merged_coils_Gx,
                              (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N
        coil_Gy_mat = np.reshape(merged_coils_Gy,
                              (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N
        coil_Gz_mat = np.reshape(merged_coils_Gz,
                              (n_channels, -1)).T[mask_erode_vec != 0, :]  # masked points x N
                              
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_erode_vec != 0]  # mV'
        
        unshimmed_Gx_vec = np.reshape(np.gradient(self.unshimmed,axis=0), (-1,))[mask_erode_vec != 0]  # mV'
        unshimmed_Gy_vec = np.reshape(np.gradient(self.unshimmed,axis=1), (-1,))[mask_erode_vec != 0]  # mV'
        unshimmed_Gz_vec = np.reshape(np.gradient(self.unshimmed,axis=2), (-1,))[mask_erode_vec != 0]  # mV'
                
        scipy_constraints = self._define_scipy_constraints()

        # Set up output currents
        currents_0 = self.get_initial_guess()

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
            stability_factor = self._criteria_func(self._initial_guess_zeros(), unshimmed_vec, np.zeros_like(coil_mat), unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, factor=1)

            currents_sp = self._scipy_minimize(currents_0, unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, scipy_constraints, factor=stability_factor)

        if not currents_sp.success:
            raise RuntimeError(f"Optimization failed due to: {currents_sp.message}")

        currents = currents_sp.x

        return currents


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
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
                                'mae': mean absolute error, 'std': standard deviation.
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
        """

        super().__init__(coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)
        self.pressure_min = pmu.min
        self.pressure_max = pmu.max
        self.initial_guess_method = 'zeros'

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

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, scipy_constraints, factor):
        """Redefined from super() since normal bounds are now constraints"""

        currents_sp = opt.minimize(self._criteria_func, currents_0,
                                   args=(unshimmed_vec, coil_mat, unshimmed_Gx_vec, unshimmed_Gy_vec, unshimmed_Gz_vec, coil_Gx_mat, coil_Gy_mat, coil_Gz_mat, factor),
                                   method='SLSQP',
                                   constraints=tuple(scipy_constraints),
                                   options={'maxiter': 30000})
        return currents_sp
