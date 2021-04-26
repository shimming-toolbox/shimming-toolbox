#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.coils.coil import Coil


class TestSequentialZSlice(object):
    def setup(self):
        # Set up unshimmed fieldmap
        num_vox = 100
        model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
        model_obj.generate_deltaB0('linear', [0.05, 0.01])
        tr = 0.025  # in s
        te = [0.004, 0.008]  # in s
        model_obj.simulate_measurement(tr, te)
        phase_meas1 = model_obj.get_phase()
        phase_e1 = phase_meas1[:, :, 0, 0]
        phase_e2 = phase_meas1[:, :, 0, 1]
        b0_map = (phase_e2 - phase_e1) / (te[1] - te[0])
        nz = 3

        # Construct synthetic field map based on a manipulation of model_obj across slices
        unshimmed = np.zeros([num_vox, num_vox, nz])
        unshimmed[:, :, 0] = b0_map
        unshimmed[:, :, 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
        unshimmed[:, :, 2] = unshimmed[:, :, 0] ** 2
        self.unshimmed = unshimmed

        # Set up spherical harmonics coil profile
        x, y, z = np.meshgrid(
            np.array(range(int(-num_vox / 2), int(num_vox / 2))),
            np.array(range(int(-num_vox / 2), int(num_vox / 2))),
            np.array(range(nz)),
            indexing='ij')
        profiles = siemens_basis(x, y, z)

        # Set up bounds for output currents
        max_coef = 5000
        min_coef = -5000
        bounds = []
        for _ in range(profiles.shape[3]):
            bounds.append((min_coef, max_coef))

        constraints = {
            "coef_sum_max": 8000,
            "coef_channel_minmax": bounds
        }

        coil = Coil(profiles, constraints)
        self.sph_coil = coil

        # Set up mask
        full_mask = shapes(unshimmed, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)
        self.mask = full_mask

    def test_zslice_lsq(self):

        # Optimize
        z_slices = np.array(range(self.sph_coil.z))
        currents = sequential_zslice(self.unshimmed, [self.sph_coil], self.mask, z_slices)

        # Calculate theoretical shimmed map
        shimmed = self.unshimmed + np.sum(currents * self.sph_coil.profiles, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_pseudo(self):

        # Optimize
        z_slices = np.array(range(self.sph_coil.z))
        currents = sequential_zslice(self.unshimmed, [self.sph_coil], self.mask, z_slices, method='pseudo_inverse')

        # Calculate theoretical shimmed map
        shimmed = self.unshimmed + np.sum(currents * self.sph_coil.profiles, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_2_coils_lsq(self):
        # TODO: Test with multiple coils using 2 different coils
        # Optimize
        z_slices = np.array(range(self.sph_coil.z))
        currents = sequential_zslice(self.unshimmed, [self.sph_coil, self.sph_coil], self.mask, z_slices)

        # Calculate theoretical shimmed map
        shimmed = self.unshimmed + np.sum(currents * np.concatenate([self.sph_coil.profiles, self.sph_coil.profiles],
                                                                    axis=3), axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_2_coils_pseudo(self):
        # TODO: Test with multiple coils using 2 different coils
        # Optimize
        z_slices = np.array(range(self.sph_coil.z))
        currents = sequential_zslice(self.unshimmed, [self.sph_coil, self.sph_coil], self.mask, z_slices,
                                     method='pseudo_inverse')

        # Calculate theoretical shimmed map
        shimmed = self.unshimmed + np.sum(currents * np.concatenate([self.sph_coil.profiles, self.sph_coil.profiles],
                                                                    axis=3), axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    # TODO: Test with a custom coil profile
    # TODO: Coils need to have the same shape, but also the same affine transfo, or resample into
    #  the same affine transfo
