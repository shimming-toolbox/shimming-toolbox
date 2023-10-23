import numpy as np
import pytest
import scipy.optimize as opt
from typing import List
import warnings

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.shim.sequencer import ShimSequencer, RealTimeSequencer, resample_mask
from shimmingtoolbox.shim.sequencer import define_slices, extend_slice, parse_slices, update_affine_for_ap_slices
from shimmingtoolbox.shim.sequencer import shim_max_intensity
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.utils import set_all_loggers
from shimmingtoolbox.optimizer.optimizer_utils import OptimizerUtils
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.coils.coil import Coil

class TestLsqOptimizer():
    def setUp(self):
        # Initialize common variables or objects needed for testing
        self.coils = []  # Initialize with appropriate Coil objects
        self.unshimmed =   # Example unshimmed array
        self.affine =   # Example affine transformation
        self.opt_criteria = 'mse'  # Choose your optimization criteria
        self.reg_factor = 0  # Choose the regularization factor

    def test_residuals_mae(self):
        # Test the _residuals_mae function
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        unshimmed_vec =   # Example unshimmed vector
        coil_mat =  # Example coil matrix
        factor = 1  # Example factor
        result = lsq_optimizer._residuals_mae(coef, unshimmed_vec, coil_mat, factor)
        self.assertIsInstance(result, float)

    def test_residuals_mse(self):
        # Test the _residuals_mse function
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        a = np.zeros(27, len(self.coils[0]))  # Example a array
        b = np.zeros(27)  # Example b array
        c = 0  # Example c value
        result = lsq_optimizer._residuals_mse(coef, a, b, c)
        self.assertIsInstance(result, float)

    def test_initial_guess_mse(self):
        # Test the _initial_guess_mse function
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        unshimmed_vec =   # Example unshimmed vector
        coil_mat =   # Example coil matrix
        factor = 1  # Example factor
        result = lsq_optimizer._initial_guess_mse(coef, unshimmed_vec, coil_mat, factor)
        self.assertIsInstance(result, float)

    def test_residuals_std(self):
        # Test the _residuals_std function
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        unshimmed_vec =   # Example unshimmed vector
        coil_mat =   # Example coil matrix
        factor =   # Example factor
        result = lsq_optimizer._residuals_std(coef, unshimmed_vec, coil_mat, factor)
        self.assertIsInstance(result, float)

    def test_residuals_mse_jacobian(self):
        # Test the _residuals_mse_jacobian function
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        a =   # Example a array
        b =   # Example b array
        c =   # Example c value
        result = lsq_optimizer._residuals_mse_jacobian(coef, a, b, c)
        self.assertIsInstance(result, np.ndarray)

    # Add similar test methods for other functions in the LsqOptimizer class

class TestPmuLsqOptimizer():
    def setUp(self):
        # Initialize common variables or objects needed for testing
        self.coils = []  # Initialize with appropriate Coil objects
        self.unshimmed =   # Example unshimmed array
        self.affine =   # Example affine transformation
        self.opt_criteria = 'mse'  # Choose your optimization criteria
        self.reg_factor = 0  # Choose the regularization factor
        self.pmu = None  # Initialize with an appropriate PmuResp object

    def test_residuals_mae(self):
        # Test the _residuals_mae function in the PmuLsqOptimizer class
        pmu_optimizer = PmuLsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, self.pmu, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        unshimmed_vec =   # Example unshimmed vector
        coil_mat =   # Example coil matrix
        factor =   # Example factor
        result = pmu_optimizer._residuals_mae(coef, unshimmed_vec, coil_mat, factor)
        self.assertIsInstance(result, float)

    def test_residuals_mse(self):
        # Test the _residuals_mse function in the PmuLsqOptimizer class
        pmu_optimizer = PmuLsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, self.pmu, reg_factor=self.reg_factor)
        coef = np.zeros(len(self.coils[0]))  # Example coefficient array
        a = np.zeros(27, len(self.coils[0]))  # Example a array
        b = np.zeros(27)  # Example b array
        c = 0  # Example c value
        result = pmu_optimizer._residuals_mse(coef, a, b, c)
        self.assertIsInstance(result, float)

    # Add similar test methods for other functions in the PmuLsqOptimizer class
    # Assert fonction filling for now
def assert_shimmed_map(nii_fieldmap, nii_anat, nii_mask, coil, currents, slices):
    unshimmed = nii_fieldmap.get_fdata()
    opt = Optimizer(coil, unshimmed, nii_fieldmap.affine)

    correction_per_channel = np.zeros(opt.merged_coils.shape + (len(slices),))
    shimmed = np.zeros(unshimmed.shape + (len(slices),))
    mask_fieldmap = np.zeros(unshimmed.shape + (len(slices),))
    
    for i_shim in range(len(slices)):
        correction_per_channel[..., i_shim] = currents[i_shim] * opt.merged_coils
        correction = np.sum(correction_per_channel[..., i_shim], axis=3, keepdims=False)
        shimmed[..., i_shim] = unshimmed + correction

        mask_fieldmap[..., i_shim] = resample_mask(nii_mask, nii_fieldmap, slices[i_shim]).get_fdata()

        sum_shimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * shimmed[..., i_shim]))
        sum_unshimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * unshimmed))

        assert sum_shimmed <= sum_unshimmed


if __name__ == '__main__':
    unittest.main()
