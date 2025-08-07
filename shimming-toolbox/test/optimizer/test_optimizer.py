#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import pytest
from scipy.special import pseudo_huber

from shimmingtoolbox.optimizer.lsq_optimizer import LsqOptimizer, PmuLsqOptimizer
from ..shim.test_sequencer import define_rt_sim_inputs, create_constraints, create_coil

nif_rt_fieldmap, nif_rt_target, nif_mask_rt_static, nif_mask_rt_riro, slices_rt, pmu_rt, coil_rt = \
    define_rt_sim_inputs()


class TestResiduals:
    def setup_method(self):
        # Set the dimensions of the matrices
        nx, ny, nz = 5, 5, 5
        nb_channels = 8

        # Soft mask
        mask = np.zeros((nx, ny, nz))
        mask[2, 2, 2] = 1.0
        # Set adjacent voxels to 0.5
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if abs(dx) + abs(dy) + abs(dz) == 1:
                        mask[2 + dx, 2 + dy, 2 + dz] = 0.5
        # Linearize the mask and filter out zeros
        mask_vec = mask.reshape((-1,))
        self.mask_coefficients = mask_vec[mask_vec != 0]

        # Coil
        coil_affine = np.eye(4)
        self.coil = create_coil(nx, ny, nz,
                                create_constraints(np.inf, -np.inf, np.inf, n_channels=nb_channels),
                                coil_affine, n_channel=nb_channels)

        # Coil matrix
        merged_coils = np.random.rand(nx, ny, nz, nb_channels)
        merged_coils_transposed = np.transpose(merged_coils, axes=(3, 0, 1, 2))
        merged_coils_reshaped = np.reshape(merged_coils_transposed, (nb_channels, -1))
        self.coil_mat = merged_coils_reshaped[:, mask_vec != 0].T

        # Unshimmed vector
        self.unshimmed = np.random.rand(nx, ny, nz)
        self.unshimmed_vec = np.reshape(self.unshimmed, (-1,))[mask_vec != 0]

        # Coefficients
        self.coef = np.random.rand(nb_channels)

        # Regularisation vector
        self.reg_vector = np.ones(nb_channels)

        # Refularisation factor
        self.factor = 1

        # Delta variable for ps_huber residuals
        self.delta = None

    def test_res_mae(self):
        opt = LsqOptimizer(opt_criteria='mae',
                           coils=[self.coil],
                           unshimmed=self.unshimmed,
                           affine=np.eye(4))
        opt.mask_coefficients = self.mask_coefficients
        opt.reg_vector = self.reg_vector
        res = opt._residuals_mae(self.coef, self.unshimmed_vec, self.coil_mat, self.factor)

        shimmed_vec = self.unshimmed_vec + self.coil_mat @ self.coef
        mae = np.sum(self.mask_coefficients * np.abs(shimmed_vec)) / np.sum(self.mask_coefficients)
        ref = mae / self.factor + np.square(self.coef).dot(self.reg_vector)

        assert np.isclose(res, ref, atol=1e-12)

    def test_res_initial_guess_mse(self):
        opt = LsqOptimizer(opt_criteria='mse',
                           coils=[self.coil],
                           unshimmed=self.unshimmed,
                           affine=np.eye(4))
        opt.mask_coefficients = self.mask_coefficients
        opt.reg_vector = self.reg_vector
        res = opt._initial_guess_mse(self.coef, self.unshimmed_vec, self.coil_mat, self.factor)

        shimmed_vec = self.unshimmed_vec + self.coil_mat @ self.coef
        mse = np.sum(self.mask_coefficients * np.square(shimmed_vec)) / np.sum(self.mask_coefficients)
        ref = mse / self.factor + np.square(self.coef).dot(self.reg_vector)

        assert np.isclose(res, ref, atol=1e-12)

    def test_res_mse(self):
        opt = LsqOptimizer(opt_criteria='mse',
                           coils=[self.coil],
                           unshimmed=self.unshimmed,
                           affine=np.eye(4))
        opt.mask_coefficients = self.mask_coefficients
        opt.reg_vector = self.reg_vector
        a, b, c = opt.get_quadratic_term(self.unshimmed_vec, self.coil_mat, self.factor)
        res = opt._residuals_mse(self.coef, a, b, c)

        shimmed_vec = self.unshimmed_vec + self.coil_mat @ self.coef
        mse = np.sum(self.mask_coefficients * np.square(shimmed_vec)) / np.sum(self.mask_coefficients)
        ref = mse / self.factor + np.square(self.coef).dot(self.reg_vector)

        assert np.isclose(res, ref, atol=1e-12)

    def test_res_rmse(self):
        opt = LsqOptimizer(opt_criteria='rmse',
                           coils=[self.coil],
                           unshimmed=self.unshimmed,
                           affine=np.eye(4))
        opt.mask_coefficients = self.mask_coefficients
        opt.reg_vector = self.reg_vector
        res = opt._residuals_rmse(self.coef, self.unshimmed_vec, self.coil_mat, self.factor)

        shimmed_vec = self.unshimmed_vec + self.coil_mat @ self.coef
        mse = np.sum(self.mask_coefficients * np.square(shimmed_vec)) / np.sum(self.mask_coefficients)
        ref = np.sqrt(mse) / self.factor + np.square(self.coef).dot(self.reg_vector)

        assert np.isclose(res, ref, atol=1e-12)

    def test_res_ps_huber(self):
        opt = LsqOptimizer(opt_criteria='ps_huber',
                           coils=[self.coil],
                           unshimmed=self.unshimmed,
                           affine=np.eye(4))
        opt.mask_coefficients = self.mask_coefficients
        opt.reg_vector = self.reg_vector
        res = opt._residuals_ps_huber(self.coef, self.unshimmed_vec, self.coil_mat, self.factor)

        shimmed_vec = self.unshimmed_vec + self.coil_mat @ self.coef
        if self.delta is None:
            self.delta = np.percentile(np.abs(shimmed_vec), 90)
        mpsh = np.sum(self.mask_coefficients * pseudo_huber(self.delta, shimmed_vec)) / np.sum(self.mask_coefficients)
        ref = mpsh / self.factor + np.square(self.coef).dot(self.reg_vector)

        assert np.isclose(res, ref, atol=1e-12)


@pytest.mark.parametrize(
    "nif_fieldmap,nif_target,nif_mask_static,nif_mask_riro,slices,pmu,coil", [(
            nif_rt_fieldmap,
            nif_rt_target,
            nif_mask_rt_static,
            nif_mask_rt_riro,
            slices_rt,
            pmu_rt,
            coil_rt
    )]
)
class TestPmuLsqOptimizer:
    def test_define_rt_bounds(self, nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro, slices, pmu,
                              coil):

        constraints = create_constraints(300, -700, 2000, 3)
        new_coil = create_coil(nif_fieldmap.shape[0], nif_fieldmap.shape[1], 3, constraints, nif_fieldmap.affine, 3)

        pmu_lsq_opt = PmuLsqOptimizer([new_coil], nif_fieldmap.data.mean(axis=3), nif_fieldmap.affine, 'ps_huber', pmu)
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
