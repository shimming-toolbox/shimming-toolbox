#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
import warnings

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer


class LsqOptimizer(Optimizer):
    """ Optimizer object that stores coil profiles and optimizes an unshimmed volume given a mask.
        Use optimize(args) to optimize a given mask. The algorithm uses a least squares solver to find the best shim.
        It supports bounds for each channel as well as a bound for the absolute sum of the channels.
    """

    def _residuals(self, coef, unshimmed_vec, coil_mat):
        """
        Objective function to minimize
        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                (axis 0 must align with unshimmed_vec)

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector
        """
        if unshimmed_vec.shape[0] != coil_mat.shape[0]:
            ValueError(f"Unshimmed ({unshimmed_vec.shape}) and coil ({coil_mat.shape} arrays do not align on axis 0")

        return np.sum(np.abs(unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)))

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

        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N --> masked points, N
        coil_mat = np.reshape(np.transpose(self.merged_coils, axes=(3, 0, 1, 2)),
                              (n_channels, -1)).T[mask_vec != 0, :]  # masked points x N
        unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_vec != 0]  # mV'

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

        # Set up output currents
        currents_0 = self.initial_guess_mean_bounds()

        # Optimize
        # When clipping to bounds, scipy raises a warning. Since this can be frequent for our purposes, we ignore that
        # warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="Values in x were outside bounds during a "
                                                      "minimize step, clipping to bounds",
                                    category=RuntimeWarning,
                                    module='scipy')

            currents_sp = opt.minimize(self._residuals, currents_0,
                                       args=(unshimmed_vec, coil_mat),
                                       method='SLSQP',
                                       bounds=self.merged_bounds,
                                       constraints=tuple(constraints),
                                       options={'maxiter': 500})

        if not currents_sp.success:
            raise RuntimeError(f"Optimization failed due to: {currents_sp.message}")

        currents = currents_sp.x

        return currents
