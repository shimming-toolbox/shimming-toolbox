#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import nibabel as nib
import pytest
import os

from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.optimizer.lsq_optimizer import PmuLsqOptimizer
from ..shim.test_sequencer import define_rt_sim_inputs, create_constraints, create_coil, create_unshimmed_affine
from shimmingtoolbox.masking.mask_utils import basic_softmask
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__

nii_rt_fieldmap, json_rt_data, nii_rt_anat, nii_mask_rt_static, nii_mask_rt_riro, slices_rt, pmu_rt, coil_rt = \
    define_rt_sim_inputs()


@pytest.mark.parametrize(
    "nii_fieldmap,json_data,nii_anat,nii_mask_static,nii_mask_riro,slices,pmu,coil", [(
            nii_rt_fieldmap,
            json_rt_data,
            nii_rt_anat,
            nii_mask_rt_static,
            nii_mask_rt_riro,
            slices_rt,
            pmu_rt,
            coil_rt
    )]
)
class TestPmuLsqOptimizer:
    def test_define_rt_bounds(self, nii_fieldmap, json_data, nii_anat, nii_mask_static, nii_mask_riro, slices, pmu,
                              coil):

        constraints = create_constraints(300, -700, 2000, 3)
        new_coil = create_coil(nii_fieldmap.shape[0], nii_fieldmap.shape[1], 3, constraints, nii_fieldmap.affine, 3)

        pmu_lsq_opt = PmuLsqOptimizer([new_coil], nii_fieldmap.get_fdata().mean(axis=3), nii_fieldmap.affine, 'ps_huber', pmu)
        pmu_lsq_opt.pressure_min = 0
        pmu_lsq_opt.pressure_max = 4095
        pmu_lsq_opt.pressure_mean = 3000
        rt_bounds = pmu_lsq_opt.define_rt_bounds()

        test_coefs = np.linspace(rt_bounds[0][1], rt_bounds[0][0], 1000)
        test_coefs_min = (pmu_lsq_opt.pressure_min - pmu_lsq_opt.pressure_mean) * test_coefs
        test_coefs_max = (pmu_lsq_opt.pressure_max - pmu_lsq_opt.pressure_mean) * test_coefs
        original_bounds = constraints['coef_channel_minmax']['coil'][0]
        if (np.any(test_coefs_min < original_bounds[0]) or np.any(test_coefs_min > original_bounds[1])
                or np.any(test_coefs_max < original_bounds[0]) or np.any(test_coefs_max > original_bounds[1])):
            assert False


def generate_dummy_inputs():
    # Dimensions of the unshimmed volume
    nx, ny, nz = 30, 30, 6  # Must be multiple of 3

    # Create the unshimmed volume
    unshimmed = np.random.rand(nx, ny, nz)
    unshimmed_affine = create_unshimmed_affine()

    # Create the coil profiles
    coil_affine = unshimmed_affine * 2
    coil_affine[3, 3] = 1
    coil = create_coil(nx, ny, nz, create_constraints(1000, -1000, 2000), coil_affine)

    # Create a binary mask
    anat = np.ones((nx, ny, nz))
    static_binmask = shapes(anat, 'cube', len_dim1=10, len_dim2=10, len_dim3=nz)
    nii_binmask = nib.Nifti1Image(static_binmask.astype(np.uint8), unshimmed_affine)

    # Create a softmask
    path_sct_binmask = os.path.join(__dir_testing__, 'tmp', 'binmask.nii.gz')
    os.makedirs(os.path.dirname(path_sct_binmask), exist_ok=True)
    nii_binmask.to_filename(path_sct_binmask)
    softmask = basic_softmask(path_sct_binmask, 3, 0.5)
    os.remove(path_sct_binmask)
    os.rmdir(os.path.join(__dir_testing__, 'tmp'))

    return unshimmed, unshimmed_affine, coil, static_binmask, softmask

# def test_get_coil_mat_and_unshimmed(self):
#     # Get the input data
#     unshimmed, unshimmed_affine, coil, static_binmask, softask = self.generate_dummy_inputs()

#     # Run the optimizer
#     optimizer = Optimizer([coil], unshimmed, unshimmed_affine)
#     bin_coil_mat, bin_unshimmed_vec = optimizer.get_coil_mat_and_unshimmed(static_binmask)
#     soft_coil_mat, soft_unshimmed_vec = optimizer.get_coil_mat_and_unshimmed(softask)

#     return bin_coil_mat, bin_unshimmed_vec, soft_coil_mat, soft_unshimmed_vec

        # Check the dimensions of the output
        # assert coil_mat.ndim == 2, "The coil matrix must be two-dimensional."
        # assert unshimmed_vec.ndim == 1, "The unshimmed vector must be one-dimensional."
        # assert coil_mat.shape[0] == unshimmed_vec.shape[0], "The number of masked points must be the same in both arrays."
        # assert unshimmed_vec.shape[0] == (softmask > 0).sum(), "The number of masked points must match the number of points in the softmask."


if __name__ == "__main__" :
    unshimmed, unshimmed_affine, coil, static_binmask, softmask = generate_dummy_inputs()
    optimizer = Optimizer([coil], unshimmed, unshimmed_affine)
    bin_profiles = optimizer.optimize(static_binmask)
    soft_profiles = optimizer.optimize(softmask)
    print(bin_profiles)
    print(soft_profiles)
