#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import os

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coordinates import generate_meshgrid


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
        # self.un_affine = np.array([[0., 0.,    3., -3.61445999],
        #                            [-2.91667008, 0., 0., 101.76699829],
        #                            [0., 2.91667008, 0., -129.85464478],
        #                            [0., 0., 0., 1.]])
        self.un_affine = np.eye(4) * 2
        self.un_affine[3, 3] = 1

        # Set up spherical harmonics coil profile
        affine = np.eye(4) * 4
        affine[3, 3] = 1
        x, y, z = generate_meshgrid((150, 150, nz), affine)
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

        coil = Coil(profiles, affine, constraints)
        self.sph_coil = coil

        # Set up mask
        full_mask = shapes(unshimmed, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)
        self.mask = full_mask

    def test_zslice_lsq(self):
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))
        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil], self.mask, z_slices)

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil])
        coils, _ = opt.merge_coils(self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_pseudo(self):
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))
        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil], self.mask, z_slices,
                                     method='pseudo_inverse')

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil])
        coils, _ = opt.merge_coils(self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    def test_zslice_2_coils_lsq(self):
        # TODO: Test with multiple coils using 2 different coils
        # Optimize
        z_slices = np.array(range(self.unshimmed.shape[2]))
        currents = sequential_zslice(self.unshimmed, self.un_affine, [self.sph_coil, self.sph_coil], self.mask,
                                     z_slices)

        # Calculate theoretical shimmed map
        opt = Optimizer([self.sph_coil, self.sph_coil])
        coils, _ = opt.merge_coils(self.unshimmed, self.un_affine)
        shimmed = self.unshimmed + np.sum(currents * coils, axis=3, keepdims=False)

        for i_slice in z_slices:
            sum_shimmed = np.sum(np.abs(self.mask[:, :, i_slice] * shimmed[:, :, i_slice]))
            sum_unshimmed = np.sum(np.abs(self.mask[:, :, i_slice] * self.unshimmed[:, :, i_slice]))
            print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
            assert sum_shimmed < sum_unshimmed

    # def test_zslice_2_different_coils_lsq(self):
    #     # unshimmed
    #     nii_unshimmed = nib.load(os.path.join(__dir_testing__, "realtime_zshimming_data/nifti/sub-example/fmap/sub-example_fieldmap.nii.gz"))
    #     affine = nii_unshimmed.affine
    #     unshimmed = nii_unshimmed.get_fdata()[..., 0]
    #
    #     # Set up custom coil
    #     fname_custom_coil = "/Users/alex/Documents/School/Polytechnique/Master/project/shimming-toolbox/temp_folder/" \
    #                         "fig_coil_profiles.nii.gz"
    #     nii_custom_coil = nib.load(fname_custom_coil)
    #
    #     # Set up bounds for output currents
    #     max_coef = 3
    #     min_coef = -3
    #     bounds = []
    #     for _ in range(nii_custom_coil.shape[3]):
    #         bounds.append((min_coef, max_coef))
    #
    #     constraints = {
    #         "coef_sum_max": 20,
    #         "coef_channel_minmax": bounds
    #     }
    #
    #     custom_coil = Coil(nii_custom_coil.get_fdata(), nii_custom_coil.affine, constraints)
    #
    #     # Optimize
    #     z_slices = np.array(range(unshimmed.shape[2]))
    #     mask = shapes(unshimmed, 'cube', len_dim1=40, len_dim2=40, len_dim3=1)
    #     currents = sequential_zslice(unshimmed, affine, [self.sph_coil, custom_coil], mask, z_slices)
    #
    #     # Calculate theoretical shimmed map
    #     opt = Optimizer([self.sph_coil, custom_coil])
    #     coils, _ = opt.merge_coils(unshimmed, affine)
    #     shimmed = unshimmed + np.sum(currents * coils, axis=3, keepdims=False)
    #
    #     for i_slice in z_slices:
    #         sum_shimmed = np.sum(np.abs(mask[:, :, i_slice] * shimmed[:, :, i_slice]))
    #         sum_unshimmed = np.sum(np.abs(mask[:, :, i_slice] * unshimmed[:, :, i_slice]))
    #         print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: {currents[i_slice, :]}")
    #         assert sum_shimmed < sum_unshimmed
    # TODO: Test with a custom coil profile
