#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coil import Coil

ListCoil = List[Coil]


class OptimizerUtils(Optimizer):

    """ Optimizer object that stores different useful functions and parameter for different optimization

        Attributes:
            initial_guess_method (string) : String indicating how to find the first guess for the optimization
            initial_coefs (np.ndarray): Initial guess that will be used in the optimization
            reg_vector (np.ndarray) : Vector used to make the regularization in the optimization

    """

    def __init__(self, coils: ListCoil, unshimmed, affine, initial_guess_method, reg_factor=0):
        """
        Initializes coils according to input list of Coil

        Args:
            coils (ListCoil): List of Coil objects containing the coil profiles and related constraints
            unshimmed (np.ndarray): 3d array of unshimmed volume
            affine (np.ndarray): 4x4 array containing the affine transformation for the unshimmed array
            reg_factor (float): Regularization factor for the current when optimizing. A higher coefficient will
                                penalize higher current values while a lower factor will lower the effect of the
                                regularization. A negative value will favour high currents (not preferred).
        """
        super().__init__(coils, unshimmed, affine)
        self.initial_guess_method = initial_guess_method
        self.initial_coefs = None
        reg_factor_channel = np.array([max(np.abs(bound)) for bound in self.merged_bounds])
        self.reg_vector = reg_factor / (len(reg_factor_channel) * reg_factor_channel)

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
            mask (np.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).

        Returns:
            np.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function.
                           The shape of the array returned has shape corresponding to the total number of channels
        """

        coil_mat, unshimmed_vec = self.get_coil_mat_and_unshimmed(mask)
        # Set up output currents
        currents_0 = self.get_initial_guess()
        # If what to shim is already 0s
        if np.all(unshimmed_vec == 0):
            return np.zeros(np.shape(currents_0))
        currents = self._get_currents(unshimmed_vec, coil_mat, currents_0)

        return currents
