import numpy as np
import pytest
import scipy.optimize as opt
from typing import List
import warnings

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.pmu import PmuResp
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
unshimmed =   
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
            MAE_base += np.abs(unshimmed[i]+(coils@coef)[i])/n
        lsq_optimizer = LsqOptimizer(coil, unshimmed, affine, opt_criteria, reg_factor=reg_factor)
        result = lsq_optimizer._residuals_mae(coef, unshimmed, coil, factor)
        assert_result(MAE_base, result)

    def test_residuals_mse(self):
        # Test the _residuals_mse function
        MSE_base = 0
        coef = 
        for i in range(n):
            MSE_base += ((unshimmed[i]+(coils@coef)[i])**2)/n
        lsq_optimizer = LsqOptimizer(coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)  
        result = lsq_optimizer._residuals_mse(coef, a, b, c)
        assert_result(MSE_base,result)

    def test_residuals_std(self):
        # Test the _residuals_std function
        STD_base = 0
        coef =
        Mean = np.mean(unshimmed+coil@coef)
        for i in range(n):
            STD_base += ((unshimmed[i]+(coils@coef)[i]-Mean)**2)/n
        STD_base = np.sqrt(STD_base)
        lsq_optimizer = LsqOptimizer(coils, unshimmed, affine, opt_criteria, reg_factor=reg_factor)
        result = lsq_optimizer._residuals_std(coef, unshimmed_vec, coil_mat, factor)
        assert_result(STD_base,result)

    def test_residuals_mse_jacobian(self):
        # Test the _residuals_mse_jacobian function
        
        Jacobian_base = 
        lsq_optimizer = LsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, reg_factor=self.reg_factor)
        result = lsq_optimizer._residuals_mse_jacobian(coef, a, b, c)
        self.assertIsInstance(result, np.ndarray)

    

class TestPmuLsqOptimizer():
    
    def test_residuals_mae(self):
        # Test the _residuals_mae function in the PmuLsqOptimizer class
        result = PmuLsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, self.pmu, reg_factor=self.reg_factor)
        MAE_base = 0
        coef = 
        for i in range(n):
            MAE_base += np.abs(unshimmed[i]+(coils@coef)[i])/n
        assert_result(MAE_base, result)

    def test_residuals_mse(self):
        # Test the _residuals_mse function in the PmuLsqOptimizer class
        result = PmuLsqOptimizer(self.coils, self.unshimmed, self.affine, self.opt_criteria, self.pmu, reg_factor=self.reg_factor)
        MSE_base = 0
        coef = 
        for i in range(n):
            MSE_base += ((unshimmed[i]+(coils@coef)[i])**2)/n
        assert_result(MAE_base, result)

    
    
    # Assert fonction filling for now
def assert_result(Base,result):

    if isinstance(Base, (float, int)) and isinstance(result, (float, int)):
        return abs(Base - result) <= tolerance
    elif isinstance(Base, np.ndarray) and isinstance(result, np.ndarray) and base.shape == result.shape:
        return np.all(np.abs(Base - result) <= tolerance)
    else:
        raise ValueError("Incompatible types for 'base' and 'result'")