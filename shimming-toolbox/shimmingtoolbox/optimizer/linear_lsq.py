# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import lsq_linear

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer


class LinearLsqOptimizer(Optimizer):
    def _get_currents(self, unshimmed_vec, coil_mat):
        res = lsq_linear(coil_mat, -1 * unshimmed_vec,
                         bounds=np.array(self.merged_bounds_off_channels).T,
                         method='trf',
                         lsmr_tol='auto',
                         max_iter=10000)
        currents = res.x
        return currents
