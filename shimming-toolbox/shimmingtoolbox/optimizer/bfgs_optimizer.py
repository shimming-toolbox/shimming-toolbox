#!/usr/bin/python3
# -*- coding: utf-8 -*-

from scipy import optimize as opt

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer, PmuLsqOptimizer


class BFGSOpt(LsqOptimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize (args) to optimize a given mask. The algorithm uses a gradient based solver (L-BFGS-B)
        to find the best shim. It supports bounds for each shim channel.
    """

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        """ Minimize the criteria function using scipy's minimize function. """

        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='L-BFGS-B',
                                       bounds=self.merged_bounds,
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        elif self.opt_criteria == 'mse_signal_recovery':
            a, b, c = self.get_quadratic_term_grad(unshimmed_vec, coil_mat, factor)
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='L-BFGS-B',
                                       bounds=self.merged_bounds,
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        else:
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(unshimmed_vec, coil_mat, factor),
                                       method='L-BFGS-B',
                                       bounds=self.merged_bounds,
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        return currents_sp


class PmuBFGSOpt(PmuLsqOptimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize (args) to optimize a given mask. The algorithm uses a gradient based solver (L-BFGS-B)
        to find the best shim. It supports bounds for each shim channel.
    """

    def _scipy_minimize(self, currents_0, unshimmed_vec, coil_mat, scipy_constraints, factor):
        """ Minimize the criteria function using scipy's minimize function. """

        if self.opt_criteria == 'mse':
            a, b, c = self.get_quadratic_term(unshimmed_vec, coil_mat, factor)
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(a, b, c),
                                       method='L-BFGS-B',
                                       bounds=self.rt_bounds,
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        else:
            currents_sp = opt.minimize(self._criteria_func, currents_0,
                                       args=(unshimmed_vec, coil_mat, factor),
                                       method='L-BFGS-B',
                                       bounds=self.rt_bounds,
                                       jac=self._jacobian_func,
                                       options={'maxiter': 10000, 'ftol': 1e-9})

        return currents_sp
