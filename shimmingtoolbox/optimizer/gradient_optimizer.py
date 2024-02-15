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
