#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer


class LSQ_Optimizer(Optimizer):

    def _residuals(self, coef, unshimmed_vec, coil_mat):
        """
        Objective function to minimize
        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coils_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils (axis 0 must align with unshimemd_vec)

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector

        """
        self._error_if(unshimmed_vec.shape[0] != coil_mat.shape[0], f'Unshimmed ({unshimmed_vec.shape}) and coil ({coil_mat.shape}) arrays do not align on axis 0')
        return unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)

    def optimize(self, unshimmed, mask, mask_origin=(0, 0, 0), bounds=None):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            unshimmed (numpy.ndarray): 3D B0 map
            mask (numpy.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).
            mask_origin (tuple): Mask origin if mask volume does not cover unshimmed volume
            bounds (list): List of ``(min, max)`` pairs for each coil channels. None
               is used to specify no bound.

        Returns:
            numpy.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function
                           (coils.size)
        """

        # Check for sizing errors
        self._check_sizing(unshimmed, mask, mask_origin=mask_origin, bounds=bounds)

        m_x, m_y, m_z = mask_origin
        m_X, m_Y, m_Z = mask.shape
        m_V = m_X * m_Y * m_Z
        mask_vec = mask.reshape((m_V,))

        # Set up optimization vectors
        coil_mat = np.reshape(np.transpose(self.coils[m_x:m_x+m_X, m_y:m_y+m_Y, m_z:m_z+m_Z], axes=(3, 0, 1, 2)), (self.N, m_V)).T[mask_vec != 0, :] # m_V' x N
        unshimmed_vec = np.reshape(unshimmed[m_x:m_x+m_X, m_y:m_y+m_Y, m_z:m_z+m_Z], (m_V,))[mask_vec != 0] # m_V'

        # Set up output currents and optimize
        currents_0 = np.zeros(self.N)
        currents_sp = opt.least_squares(self._residuals, currents_0, args=(unshimmed_vec, coil_mat), bounds=np.array(bounds).T)

        currents = currents_sp.x

        return currents
