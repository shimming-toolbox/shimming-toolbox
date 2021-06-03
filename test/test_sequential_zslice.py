#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coordinates import generate_meshgrid


class TestSequentialZSlice(object):
    def setup(self):
        # Set up 2-dimensional unshimmed fieldmaps
        num_vox = 100
        model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
        model_obj.generate_deltaB0('linear', [0.0, 20])
        tr = 0.025  # in s
        te = [0.004, 0.008]  # in s
        model_obj.simulate_measurement(tr, te)
        phase_meas1 = model_obj.get_phase()
        phase_e1 = phase_meas1[:, :, 0, 0]
        phase_e2 = phase_meas1[:, :, 0, 1]
        b0_map = ((phase_e2 - phase_e1) / (te[1] - te[0])) / (2 * np.pi)

        # Construct a 3-dimensional synthetic field map by stacking different z-slices along the 3rd dimension. Each
        # slice is subjected to a manipulation of model_obj across slices (e.g. rotation, squared) in order to test
        # various shim configurations.
        nz = 3  # Must be multiple of 3
        unshimmed = np.zeros([num_vox, num_vox, nz])
        for i_n in range(nz // 3):
            unshimmed[:, :, 3 * i_n] = b0_map
            unshimmed[:, :, (3 * i_n) + 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
            unshimmed[:, :, (3 * i_n) + 2] = unshimmed[:, :, 0] ** 2
        self.unshimmed = unshimmed
        self.un_affine = np.array([[0., 0., 3., 1],
                                   [-2.91667008, 0., 0., 2],
                                   [0., 2.91667008, 0., 3],
                                   [0., 0., 0., 1.]])

        # Set up spherical harmonics coil profile
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        x, y, z = generate_meshgrid((150, 150, nz+10), affine)
        profiles = siemens_basis(x, y, z)

        # Set up bounds for output currents
        max_coef = 1000
        min_coef = -1000
        bounds = []
        for _ in range(profiles.shape[3]):
            bounds.append((min_coef, max_coef))

        self.constraints = {
            "coef_sum_max": 2000,
            "coef_channel_minmax": bounds
        }

        # Define coil1
        coil = Coil(profiles, affine, self.constraints)
        self.sph_coil = coil

        # Define coil2
        affine = np.eye(4) * 0.75
        affine[3, 3] = 1
        x, y, z = generate_meshgrid((150, 120, 5), affine)
        profiles = siemens_basis(x, y, z)
        constraints = {
            "coef_sum_max": 1500,
            "coef_channel_minmax": tuple(np.array(bounds) / 2)
        }
        self.sph_coil_2 = Coil(profiles, affine, constraints)

        # Set up mask
        mask = shapes(unshimmed, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)
        self.mask = mask

    def test_zslice_lsq(self):
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))
        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil], self.mask, z_slices,
                                     method='least_squares')

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil], self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * opt.merged_coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            # print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_pseudo(self):
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))
        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil], self.mask, z_slices,
                                     method='pseudo_inverse')

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil], self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * opt.merged_coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            # print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_2_coils_lsq(self):
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))

        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil, self.sph_coil_2], self.mask,
                                     z_slices, method='least_squares')

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil, self.sph_coil_2], self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * opt.merged_coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            # print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    # def test_speed_huge_matrix(self):
    #     # Create 1 huge coil which essentially is siemens basis concatenated 4 times
    #     coils = [self.sph_coil, self.sph_coil, self.sph_coil, self.sph_coil]
    #
    #     coil_profiles_list = []
    #     bounds = []
    #
    #     for coil in coils:
    #         # Concat coils and bounds
    #         coil_profiles_list.append(coil.profile)
    #         for a_bound in coil.coef_channel_minmax:
    #          bounds.append(a_bound)
    #
    #     coil_profiles = np.concatenate(coil_profiles_list, axis=3)
    #     constraints = self.constraints
    #     constraints['coef_channel_minmax'] = bounds
    #     huge_coil = Coil(coil_profiles, self.sph_coil.affine, constraints)
    #
    #     z_slices = np.array(range(self.unshimmed.shape[2]))
    #     currents = sequential_zslice(self.unshimmed, self.un_affine, [huge_coil], self.mask, z_slices)
    #
    #     # Calculate theoretical shimmed map
    #     opt = Optimizer([huge_coil], self.unshimmed, self.un_affine)
    #     shimmed = self.unshimmed + np.sum(currents * opt.merged_coils, axis=3, keepdims=False)
    #
    #     for i_slice in z_slices:
    #         sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
    #         sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
    #         print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_slice, :]}")
    #         assert sum_shimmed <= sum_unshimmed
