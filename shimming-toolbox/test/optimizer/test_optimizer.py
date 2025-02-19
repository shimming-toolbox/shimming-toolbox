#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import pytest

from shimmingtoolbox.optimizer.lsq_optimizer import PmuLsqOptimizer
from ..shim.test_sequencer import define_rt_sim_inputs, create_constraints, create_coil

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
