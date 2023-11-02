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

def create_coil(n_x, n_y, n_z, constraints, coil_affine, n_channel=8):
    # Set up spherical harmonics coil profile
    mesh_x, mesh_y, mesh_z = generate_meshgrid((n_x, n_y, n_z), coil_affine)
    profiles = siemens_basis(mesh_x, mesh_y, mesh_z)

    # Define coil1
    coil = Coil(profiles[..., :n_channel], coil_affine, constraints)
    return coil

def create_constraints(max_coef, min_coef, sum_coef, n_channels=8):
    # Set up bounds for output currents
    bounds = []
    for _ in range(n_channels):
        bounds.append((min_coef, max_coef))

    constraints = {
        "name": "test",
        "coef_sum_max": sum_coef,
        "coef_channel_minmax": bounds
    }
    return constraints

# Initialize common variables or objects needed for testing
constraints = create_constraints(max_coef,min_coef,sum_coef,n_channels=8) # Initialize constraints
coils = create_coil(n_x,n_y,n_z,constraints, coil_affine, n_channels=8)  # Initialize coil
unshimmed =   # Initialize unshimmed vector
# self.affine =   
# self.opt_criteria = 'mse'  
reg_factor = 0  # Choose the regularization factor
n = len(unshimmed)
factor =  # inv_factor
# Initialize (a,b,c) for the MSE 
a = np.transpose(coil)@coil*factor
b = 2*factor*np.transpose(unshimmed)@coil
c = factor*np.transpose(unshimmed)@unshimmed

class TestLsqOptimizer():

    def test_residuals_mae(self):
        # Test the _residuals_mae function
        MAE_base = 0
        coef = 
        for i in range(n):
            MAE_base += np.abs(unshimmed[i]+(coil@coef)[i])/n
        lsq_optimizer = LsqOptimizer(coil, unshimmed, affine, opt_criteria, reg_factor=reg_factor)
        result = lsq_optimizer._residuals_mae(coef, unshimmed, coil, factor)
        assert_result(MAE_base, result)

    def test_residuals_mse(self):
        # Test the _residuals_mse function
        MSE_base = 0
        coef = 
        for i in range(n):
            MSE_base += ((unshimmed[i]+(coil@coef)[i])**2)/n
        lsq_optimizer = LsqOptimizer(coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)  
        result = lsq_optimizer._residuals_mse(coef, a, b, c)
        assert_result(MSE_base,result)

    def test_residuals_std(self):
        # Test the _residuals_std function
        STD_base = 0
        coef =
        Mean = np.mean(unshimmed+coil@coef)
        for i in range(n):
            STD_base += ((unshimmed[i]+(coil@coef)[i]-Mean)**2)/n
        STD_base = np.sqrt(STD_base)
        lsq_optimizer = LsqOptimizer(coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)
        result = lsq_optimizer._residuals_std(coef, unshimmed_vec, coil_mat, factor)
        assert_result(STD_base,result)

    def test_residuals_mse_jacobian(self):
        # Test the _residuals_mse_jacobian function
        def cost_function(unshimmed,coil,coef):
            return unshimmed+coil@x
        Jacobian_base = 
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        result = lsq_optimizer._residuals_mse_jacobian(coef, a, b, c)
        self.assertIsInstance(result, np.ndarray)

    

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

    
    
    # Assert fonction filling for now
def assert_result(Base,result):

