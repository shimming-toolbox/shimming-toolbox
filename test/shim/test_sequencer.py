#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from shimmingtoolbox.shim.sequencer import shim_sequencer
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.coils.coordinates import generate_meshgrid


def create_unshimmed():
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
    unshimmed = np.zeros([num_vox, num_vox, nz])
    for i_n in range(nz // 3):
        unshimmed[:, :, 5 * i_n] = b0_map
        unshimmed[:, :, (5 * i_n) + 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
        unshimmed[:, :, (5 * i_n) + 2] = unshimmed[:, :, 0] ** 2

    return unshimmed


def create_unshimmed_affine():
    return np.array([[0., 0., 3., 1],
                     [-2.91667008, 0., 0., 2],
                     [0., 2.91667008, 0., 3],
                     [0., 0., 0., 1.]])


def create_constraints(max_coef, min_coef, sum_coef, n_channels=8):
    # Set up bounds for output currents
    bounds = []
    for _ in range(n_channels):
        bounds.append((min_coef, max_coef))

    constraints = {
                "coef_sum_max": sum_coef,
                "coef_channel_minmax": bounds
            }
    return constraints


def create_coil(n_x, n_y, n_z, constraints, coil_affine):
    # Set up spherical harmonics coil profile
    mesh_x, mesh_y, mesh_z = generate_meshgrid((n_x, n_y, n_z), coil_affine)
    profiles = siemens_basis(mesh_x, mesh_y, mesh_z)

    # Define coil1
    coil = Coil(profiles, coil_affine, constraints)
    return coil


def create_mask(ref):
    # Set up mask
    mask = shapes(ref, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)
    return mask


nz = 3  # Must be multiple of 3
a_unshimmed = create_unshimmed()
affine = np.eye(4) * 2
affine[3, 3] = 1
coil1 = create_coil(150, 150, nz + 10, create_constraints(1000, -1000, 2000), affine)
affine = np.eye(4) * 0.75
affine[3, 3] = 1
coil2 = create_coil(150, 120, nz + 10, create_constraints(500, -500, 1500), affine)


@pytest.mark.parametrize(
    "unshimmed,un_affine,sph_coil,sph_coil2,mask", [(
        a_unshimmed,
        create_unshimmed_affine(),
        coil1,
        coil2,
        create_mask(a_unshimmed)
    )
    ]
)
class TestSequencer(object):
    def test_shim_sequencer_lsq(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        # Optimize
        z_slices = []
        for i in range(nz):
            z_slices.append((i,))
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil], mask, z_slices, method='least_squares')

        assert_results([sph_coil], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_pseudo(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        # Optimize
        z_slices = []
        for i in range(nz):
            z_slices.append((i,))
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil], mask, z_slices, method='pseudo_inverse')

        assert_results([sph_coil], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_2_coils_lsq(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):

        # Optimize
        z_slices = []
        for i in range(nz):
            z_slices.append((i,))
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil, sph_coil2], mask, z_slices, method='least_squares')

        assert_results([sph_coil, sph_coil2], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_coefs_are_none(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):

        coil = create_coil(5, 5, nz, create_constraints(None, None, None), affine)

        # Optimize
        z_slices = []
        for i in range(nz):
            z_slices.append((i,))
        currents = shim_sequencer(unshimmed, un_affine, [coil], mask, z_slices, method='least_squares')

        assert_results([coil], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_different_slices(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        # Optimize
        z_slices = [(0, 2), (1,)]
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil], mask, z_slices, method='least_squares')

        assert_results([sph_coil], unshimmed, un_affine, currents, mask, z_slices)

    # def test_speed_huge_matrix(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
    #     # Create 1 huge coil which essentially is siemens basis concatenated 4 times
    #     coils = [sph_coil, sph_coil, sph_coil, sph_coil]
    #
    #     coil_profiles_list = []
    #
    #     for coil in coils:
    #         # Concat coils and bounds
    #         coil_profiles_list.append(coil.profile)
    #
    #     coil_profiles = np.concatenate(coil_profiles_list, axis=3)
    #     constraints = create_constraints(1000, 2000, 32)
    #
    #     huge_coil = Coil(coil_profiles, sph_coil.affine, constraints)
    #
    #     z_slices = np.array(range(unshimmed.shape[2]))
    #     currents = sequential_zslice(unshimmed, un_affine, [huge_coil], mask, z_slices)
    #
    #     # Calculate theoretical shimmed map
    #     opt = Optimizer([huge_coil], unshimmed, un_affine)
    #     shimmed = unshimmed + np.sum(currents * opt.merged_coils, axis=3, keepdims=False)
    #
    #     for i_slice in z_slices:
    #         sum_shimmed = np.sum(np.abs(mask[:, :, i_slice] * shimmed[:, :, i_slice]))
    #         sum_unshimmed = np.sum(np.abs(mask[:, :, i_slice] * unshimmed[:, :, i_slice]))
    #         # print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_slice, :]}")
    #         assert sum_shimmed <= sum_unshimmed


def assert_results(coil, unshimmed, un_affine, currents, mask, z_slices):
    # Calculate theoretical shimmed map
    opt = Optimizer(coil, unshimmed, un_affine)
    shimmed = np.zeros_like(unshimmed)
    for i_shim in range(len(z_slices)):
        correction = np.sum(currents[i_shim] * opt.merged_coils, axis=3, keepdims=False)[..., z_slices[i_shim]]
        shimmed[..., z_slices[i_shim]] = unshimmed[..., z_slices[i_shim]] + correction

        sum_shimmed = np.sum(np.abs(mask[:, :, z_slices[i_shim]] * shimmed[:, :, z_slices[i_shim]]))
        sum_unshimmed = np.sum(np.abs(mask[:, :, z_slices[i_shim]] * unshimmed[:, :, z_slices[i_shim]]))
        print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_shim, :]}")
        assert sum_shimmed < sum_unshimmed
