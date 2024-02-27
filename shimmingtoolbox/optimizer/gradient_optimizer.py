#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings
from typing import List
from scipy import optimize as opt
from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer


class GradientOpt(LsqOptimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize (args) to optimize a given mask. The algorithm uses a gradient based solver (L-BFGS-B)
        to find the best shim. It supports bounds for each shim channel.
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

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, factor):
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
                            jac=self._jacobian_func if self.opt_criteria == 'mse' else None,
                            options={'maxiter': 1000})

    def _get_currents(self, unshimmed_vec, coil_mat, currents_0):

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
            if self.opt_criteria == 'mse':
                stability_factor = self._initial_guess_mse(self._initial_guess_zeros(), unshimmed_vec,
                                                           np.zeros_like(coil_mat),
                                                           factor=1)
            else:
                stability_factor = self._criteria_func(self._initial_guess_zeros(), unshimmed_vec,
                                                       np.zeros_like(coil_mat),
                                                       factor=1)
            currents_sp = self._scipy_minimize(currents_0, unshimmed_vec, coil_mat,
                                               factor=stability_factor)
        if not currents_sp.success:
            raise RuntimeError(f"Optimization failed due to: {currents_sp.message}")

        currents = currents_sp.x

        return currents
