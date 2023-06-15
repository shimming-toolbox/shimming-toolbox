#!/usr/bin/python3
# -*- coding: utf-8 -*-
import time

import numpy as np

from typing import List


from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]


import logging

from qpsolvers import solve_qp

logger = logging.getLogger(__name__)


class QuadProgOpt(Optimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a least squares solver to find the best shim.
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine)
        self._initial_guess_method = 'mean'
        self.initial_coefs = None
        if reg_factor < 0:
            raise TypeError(f" reg_ factor is negative, and would cause optimization to crash."
                            f" If you want to keep this reg_factor please use lsq_optimizer")
        reg_factor_channel = np.array([max(np.abs(bound)) for bound in self.merged_bounds])
        self.reg_vector = reg_factor / (len(reg_factor_channel) * reg_factor_channel)
        self.ineq_matrix, self.ineq_vector = self._get_linear_inequality_matrices()

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

    def _get_linear_inequality_matrices(self):
        """
        This functions returns the linear inequlity matrix and vector, that will be used in the optimization, such as
        g @ x < h, to see all details please see the PR #458

        Returns:
            (tuple) : tuple containing:
            * np.ndarray: linear inequality matrix
            * np.ndarray: linear inequality vector

        """
        n = len(self.reg_vector)
        n_coils = len(self.coils)

        sum_constraints = np.zeros([n_coils, 1])
        g_bis = np.zeros([n_coils, 2 * n])

        # First we want to create the part to verify that the sum of the currents doesn't get above a limit
        start_index = n
        for i_coil in range(n_coils):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                sum_constraints[i_coil, 0] = coil.coef_sum_max
                g_bis[i_coil, start_index: end_index] = 1
            start_index = end_index

        # Then we want to see if the currents are between a lower, and an upper bound
        lb = np.zeros([n, 1])
        lb[:, 0] = np.array([x[0] for x in self.merged_bounds])
        ub = np.zeros([n, 1])
        ub[:, 0] = np.array([x[1] for x in self.merged_bounds])

        # We create both array, but if there is no sum constraints, then there is no need to add this constraints
        if np.all(sum_constraints == 0):
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)],
                          [-np.eye(n), np.zeros([n, n])], [np.eye(n), np.zeros([n, n])],
                          [np.zeros([n, n]), -np.eye(n)],
                          [np.zeros([n, n]), np.eye(n)]])
            # dim(g) = (6n, 2n)
            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])],
                          [-lb], [ub], [np.zeros([n, 1])], [np.abs(ub) * np.ones([n, 1])]])
            # dim(h) = (6n, 1)
        else:
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)], [g_bis],
                          [-np.eye(n), np.zeros([n, n])], [np.eye(n), np.zeros([n, n])],
                          [np.zeros([n, n]), -np.eye(n)],
                          [np.zeros([n, n]), np.eye(n)]])
            # dim(g) = (6n +n_coils, 2n)
            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])], [sum_constraints],
                          [-lb], [ub], [np.zeros([n, 1])], [np.max(sum_constraints) * np.ones([n, 1])]])
            # dim(h) = (6n +n_coils, 1)

        return g, h

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

    def get_stability_factor(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to find the stability factor for the quadratic optimization

        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for least squares optimization
        """
        shimmed_vec = unshimmed_vec + coil_mat @ coef
        return (shimmed_vec).dot(shimmed_vec) / len(unshimmed_vec) / factor + np.abs(coef).dot(self.reg_vector)

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
        # # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        # #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        merged_coils_reshaped = np.reshape(np.transpose(self.merged_coils, axes=(3, 0, 1, 2)),
                                           (n_channels, -1))
        masked_points_indices = np.where(mask_vec != 0)
        coil_mat = merged_coils_reshaped[:, masked_points_indices[0]].T  # masked points x N
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[masked_points_indices[0]]  # mV'

        # Set up output currents
        currents_0 = self.get_initial_guess()
        # If what to shim is already 0s
        if np.all(unshimmed_vec == 0):
            return np.zeros(np.shape(currents_0))

        stability_factor = self.get_stability_factor(self._initial_guess_zeros(), unshimmed_vec,
                                                     np.zeros_like(coil_mat),
                                                     factor=1)

        currents = self.get_currents(unshimmed_vec, coil_mat, stability_factor, currents_0)

        return currents

    def get_currents(self, unshimmed_vec, coil_mat, factor, currents_0):
        """
        Returns the currents needed for the shimming
        Args:
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch
            currents_0 (numpy.ndarray) : Initial guess for the function

        Returns:
            np.ndarray : Vector of currents that make the better shimming

        """

        n = len(currents_0)
        initial_guess = np.zeros(2 * n)
        initial_guess[:n] = currents_0
        inv_factor = 1 / (len(unshimmed_vec) * factor)
        a = (coil_mat.T @ coil_mat) * inv_factor + np.diag(self.reg_vector)
        b = 2 * inv_factor * (unshimmed_vec @ coil_mat)

        cost_matrix = np.block([[a, np.zeros([n, n])], [np.zeros([n, n]), np.zeros([n, n])]])
        cost_matrix = 2 * cost_matrix
        cost_vector = np.zeros(2 * n)
        cost_vector[0: n] = b

        currents = solve_qp(cost_matrix, cost_vector, self.ineq_matrix, self.ineq_vector,
                            initvals=initial_guess, solver="osqp")

        if currents == 'None':
            raise TypeError(" The optimization didn't succeed, please check your parameters")

        return currents[:n]


class PmuQuadProgOpt(QuadProgOpt):
    """ Optimizer for the realtime component (riro) for this optimization:
            field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
            Unshimmed must be in units: [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

            This optimizer bounds the riro results to the coil bounds by taking the range of pressure that can be reached
            by the PMU.
        """

    def __init__(self, coils, unshimmed, affine, pmu: PmuResp, reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (numpy.ndarray): 3d array of unshimmed volume
            affine (numpy.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
        """

        super().__init__(coils, unshimmed, affine, reg_factor=reg_factor)
        self.pressure_min = pmu.min
        self.pressure_max = pmu.max
        self.ineq_matrix, self.ineq_vector = self._get_linear_inequality_matrices()
        self.initial_guess_method = 'zeros'

    def _get_linear_inequality_matrices(self):

        """
        This functions returns the linear inequlity matrix and vector, that will be used in the optimization, such as
        g @ x < h, to see all details please see the PR #458
        Redefined from QuadProgo to match the new bounds

        Returns:
            (tuple) : tuple containing:
            * np.ndarray: linear inequality matrix
            * np.ndarray: linear inequality vector

        """
        n_coils = len(self.coils)
        n = len(self.reg_vector)
        sum_constraints = np.zeros([n_coils, 1])
        ub = np.zeros([n, 1])
        lb = np.zeros([n, 1])
        g_bis = np.zeros([n_coils, 2 * n])
        start_index = n

        # First we want to create the part to verify that the sum of the currents doesn't get above a limit
        for i_coil in range(n_coils):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                pressure = np.min(np.abs(self.pressure_max), np.abs(self.pressure_min))
                sum_constraints[i_coil, 0] = coil.coef_sum_max / (coil.dim[3] * pressure)
                g_bis[i_coil, start_index: end_index] = 1
            start_index = end_index

        # Then we want to see if the currents are between a lower, and an upper bound
        delta_pressure = self.pressure_max - self.pressure_min
        for i_bound, bound in enumerate(self.merged_bounds):
            lb[i_bound, 0] = bound[0]
            ub[i_bound, 0] = bound[1]

        # We create both array, but if there is no sum constraints, then there is no need to add this constraints
        if np.all(sum_constraints == 0):

            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])],
                          [np.zeros([n, n]), -np.eye(n)],
                          [np.zeros([n, n]), np.eye(n)]])
            # dim(g) = (8n, 2n)

            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])],
                          [-lb], [-lb], [ub], [ub], [np.zeros([n, 1])], [sum_constraints * np.ones([n, 1])]])

            # dim(h) = (8n, 1)
        else:
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)], [g_bis],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])],
                          [np.zeros([n, n]), -np.eye(n)],
                          [np.zeros([n, n]), np.eye(n)]])
            # dim(g) = (8n +n_coils, 2n)

            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])], [sum_constraints],
                          [-lb], [-lb], [ub], [ub], [np.zeros([n, 1])], [np.max(sum_constraints) * np.ones([n, 1])]])
            # dim(h) = (8n +n_coils, 1)

        return g, h
