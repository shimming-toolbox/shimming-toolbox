#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import quadprog
from typing import List

from shimmingtoolbox.optimizer.optimizer_utils import OptimizerUtils
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]
logger = logging.getLogger(__name__)


class QuadProgOpt(OptimizerUtils):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a quadprog solver to find the best shim.
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def __init__(self, coils: ListCoil, unshimmed, affine, reg_factor=0, initial_guess_method='zeros'):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
            initial_guess_method (str): method to find the initial guess
        """
        if reg_factor < 0:
            raise TypeError("reg_factor is negative, and would cause optimization to crash. "
                            "If you want to keep this reg_factor please use lsq_optimizer")
        self.opt_criteria = None
        super().__init__(coils, unshimmed, affine, initial_guess_method, reg_factor)

    def _get_linear_inequality_matrices(self):
        """
        This functions returns the linear inequality matrix and vector, that will be used in the optimization, such as
        g @ x < h, to see all details please see the PR 458

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: linear inequality matrix
                * np.ndarray: linear inequality vector

        """
        n = len(self.reg_vector)
        n_coils = len(self.coils)

        sum_constraints = []
        g_bis = []

        # First we want to create the part to verify that the sum of the currents don't get above a limit
        start_index = n
        for i_coil in range(n_coils):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                sum_constraints.append(coil.coef_sum_max)
                tmp = np.zeros([2 * n])
                tmp[start_index: end_index] = 1
                g_bis.append(tmp)
            start_index = end_index
        g_bis = np.array(g_bis)
        sum_constraints = np.array(sum_constraints)[:, np.newaxis]

        # Then we want to see if the currents are between a lower, and an upper bound
        lb = np.zeros([n, 1])
        lb[:, 0] = np.array([x[0] for x in self.merged_bounds])
        ub = np.zeros([n, 1])
        ub[:, 0] = np.array([x[1] for x in self.merged_bounds])

        # We create both arrays, but if there is no sum constraints, then there is no need to add this constraint
        if np.any(sum_constraints):
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)], [g_bis],
                          [-np.eye(n), np.zeros([n, n])], [np.eye(n), np.zeros([n, n])]])
            # dim(g) = (4n + n_sum_constraints, 2n)
            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])], [sum_constraints],
                          [-lb], [ub]])
            # dim(h) = (4n + n_sum_constraints, 1)
        else:
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)],
                          [-np.eye(n), np.zeros([n, n])], [np.eye(n), np.zeros([n, n])]])
            # dim(g) = (4n, 2n)
            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])],
                          [-lb], [ub]])
            # dim(h) = (4n, 1)
        return g, h

    def get_stability_factor(self, coef, unshimmed_vec, coil_mat, factor):
        """ Objective function to find the stability factor for the quadratic optimization

        Args:
            coef (np.ndarray): 1D array of channel coefficients
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            factor (float): Devise the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            float: Residuals for quad_prog optimization
        """
        mse = np.average(np.square(unshimmed_vec + coil_mat @ coef), weights=self.mask_coefficients)
        mse_coef = mse / factor  # MSE regularized to minimize currents
        current_regularization_coef = np.square(coef).dot(self.reg_vector)

        return mse_coef + current_regularization_coef

    def _get_currents(self, unshimmed_vec, coil_mat, currents_0):
        """
        Returns the currents needed for the shimming, redefined from basic_optimizer because of the constraints
        Args:
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vec)
            currents_0 (np.ndarray) : Initial guess for the function

        Returns:
            np.ndarray : Vector of currents that make the better shimming

        """
        # This allows to scale the output for the minimize function to avoid positive directional linesearch
        stability_factor = self.get_stability_factor(self._initial_guess_zeros(), unshimmed_vec,
                                                     np.zeros_like(coil_mat),
                                                     factor=1)
        # We want to minimize 1/2 x.T @ cost_matrix @ x - cost_vector.T @ x
        cost_matrix, cost_vector = self.get_cost_matrices(currents_0, unshimmed_vec, coil_mat, stability_factor)
        # We want to get the linear inequality matrix and vector, such as g @ x < h
        ineq_matrix, ineq_vector = self._get_linear_inequality_matrices()

        # We use the quadprog solver to solve the quadratic optimization problem
        opt_result = quadprog.solve_qp(cost_matrix, -cost_vector, ineq_matrix.T, -ineq_vector[:, 0])
        currents = opt_result[0]

        if logger.level <= getattr(logging, 'DEBUG'):
            n_iter = opt_result[3][0]
            logger.debug(f"Number of iterations for quadprog: {n_iter}")

        return currents[:len(currents_0)]

    def get_cost_matrices(self, currents_0, unshimmed_vec, coil_mat, factor):
        """
        Returns the cost matrix and the cost vector to minimize 1/2 x.T @ cost_matrix @ x - cost_vector.T @ x

        Args:
            currents_0 (np.ndarray) : Initial guess for the function
            unshimmed_vec (np.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (np.ndarray): 2D flattened array (point, channel) of masked coils
                                      (axis 0 must align with unshimmed_vector)
            factor (float): Divide the result by 'factor'. This allows to scale the output for the minimize function to
                            avoid positive directional linesearch

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: 2D Cost matrix
                * np.ndarray: Cost vector

        """

        n = len(currents_0)

        initial_guess = np.zeros(2 * n)
        initial_guess[:n] = currents_0
        a, b, _ = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)
        epsilon = 1e-6
        cost_matrix = np.block([[a, np.zeros([n, n])], [np.zeros([n, n]), np.zeros([n, n])]]) + epsilon * np.eye(2*n)
        cost_matrix = 2 * cost_matrix
        cost_vector = np.zeros(2 * n)
        cost_vector[0: n] = b

        return cost_matrix, cost_vector

# TODO : Realtime softmask B0 shimming needs to be implemented
class PmuQuadProgOpt(QuadProgOpt):
    """ Optimizer for the realtime component (riro) for this optimization:
            field(i_vox) = riro(i_vox) * (acq_pressures - mean_p) + static(i_vox)
            Unshimmed must be in units: [unit_shim/unit_pressure], ex: [Hz/unit_pressure]

            This optimizer bounds the riro results to the coil bounds by taking the range of pressure that can be
            reached by the PMU.
        """

    def __init__(self, coils, unshimmed, affine, pmu: PmuResp, reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            pmu (PmuResp): PmuResp object containing the respiratory trace information.
        """

        super().__init__(coils, unshimmed, affine, reg_factor=reg_factor, initial_guess_method='zeros')
        self.pressure_min = pmu.min
        self.pressure_max = pmu.max

    def _get_linear_inequality_matrices(self):

        """
        This functions returns the linear inequality matrix and vector, that will be used in the optimization, such as
        g @ x < h, to see all details please see the PR 458
        Redefined from QuadProg to match the new bounds and constraints

        Returns:
            (tuple) : tuple containing:
                * np.ndarray: linear inequality matrix
                * np.ndarray: linear inequality vector

        """
        n_coils = len(self.coils)
        n = len(self.reg_vector)
        sum_constraints = []
        ub = np.zeros([n, 1])
        lb = np.zeros([n, 1])
        g_bis = []

        start_index = n
        # First we want to create the part to verify that the sum of the currents doesn't get above a limit
        for i_coil in range(n_coils):
            coil = self.coils[i_coil]
            end_index = start_index + coil.dim[3]
            if coil.coef_sum_max != np.inf:
                pressure = np.max(np.abs(self.pressure_max), np.abs(self.pressure_min))
                sum_constraints.append(coil.coef_sum_max / (coil.dim[3] * pressure))
                tmp = np.zeros([2 * n])
                tmp[start_index: end_index] = 1
                g_bis.append(tmp)
            start_index = end_index

        g_bis = np.array(g_bis)
        sum_constraints = np.array(sum_constraints)[:, np.newaxis]

        # Then we want to see if the currents are between a lower, and an upper bound
        delta_pressure = self.pressure_max - self.pressure_min
        for i_bound, bound in enumerate(self.merged_bounds):
            lb[i_bound, 0] = bound[0]
            ub[i_bound, 0] = bound[1]

        # We create both array, but if there is no sum constraints, then there is no need to add this constraints
        if np.any(sum_constraints):
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)], [g_bis],
                          [- delta_pressure * - np.eye(n), np.zeros([n, n])],
                          [delta_pressure * - np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])]])
            # dim(g) = (6n + n_sum_constraints, 2n)

            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])], [sum_constraints],
                          [-lb], [-lb], [ub], [ub]])
            # dim(h) = (6n + n_sum_constraints, 1)
        else:
            g = np.block([[-np.eye(n), -np.eye(n)], [np.eye(n), -np.eye(n)],
                          [- delta_pressure * - np.eye(n), np.zeros([n, n])],
                          [delta_pressure * - np.eye(n), np.zeros([n, n])],
                          [delta_pressure * np.eye(n), np.zeros([n, n])],
                          [- delta_pressure * np.eye(n), np.zeros([n, n])]])
            # dim(g) = (6n, 2n)

            h = np.block([[np.zeros([n, 1])], [np.zeros([n, 1])],
                          [-lb], [-lb], [ub], [ub]])

            # dim(h) = (6n, 1)
        return g, h
