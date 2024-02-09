#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List
from scipy import optimize as opt
from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer


class GradientOpt(LsqOptimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize (args) to optimize a given mask. The algorithm uses a gradient based solver (L-BFGS-B)
        to find the best shim. It supports bounds for each channel as well as a bound for the absolute sum
        of the channels.
    """

    def __init__(self, coils: List, unshimmed, affine, opt_criteria='mse', initial_guess_method='mean',
                 reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            opt_criteria (str): Criteria for the optimizer 'least_squares'. Supported: 'mse': mean squared error,
            'mae': mean absolute error, 'std': standard deviation, 'ps_huber': pseudo huber cost function.
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine, opt_criteria, initial_guess_method, reg_factor)

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)
            method = 'L-BFGS-B'
        elif self.opt_criteria == 'ps_huber':
            method = 'L-BFGS-B'
        else:
            method = 'L-BFGS-B'

        return opt.minimize(self._criteria_func, currents_0,
                            args=(a, b, c) if self.opt_criteria == 'mse' else (unshimmed_vec, coil_mat, factor),
                            method=method,
                            bounds=self.merged_bounds,
                            constraints=tuple(scipy_constraints),
                            jac=self._jacobian_func if self.opt_criteria == 'mse' else None,
                            options={'maxiter': 1000})
