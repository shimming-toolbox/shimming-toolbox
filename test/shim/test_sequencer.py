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

import os
import nibabel as nib
import json
from matplotlib.figure import Figure

from shimmingtoolbox.shim.sequencer import shim_realtime_pmu_sequencer
from shimmingtoolbox.shim.sequencer import shim_realtime_pmu_sequencer_anat
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.shim.sequencer import define_slices

from shimmingtoolbox.coils.coordinates import resample_from_to

DEBUG = True


def create_unshimmed():
    # Set up 2-dimensional unshimmed fieldmaps
    num_vox = 100
    model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
    model_obj.generate_deltaB0('linear', [0.025, 2])
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
        unshimmed[:, :, 3 * i_n] = b0_map
        unshimmed[:, :, (3 * i_n) + 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
        unshimmed[:, :, (3 * i_n) + 2] = unshimmed[:, :, 0] ** 2

    return unshimmed


def create_unshimmed_affine():
    # return np.array([[0., 0., 3., 1],
    #                  [-2.91667008, 0., 0., 2],
    #                  [0., 2.91667008, 0., 3],
    #                  [0., 0., 0., 1.]])
    return np.eye(4)


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


def create_coil(n_x, n_y, n_z, constraints, coil_affine, n_channel=8):
    # Set up spherical harmonics coil profile
    mesh_x, mesh_y, mesh_z = generate_meshgrid((n_x, n_y, n_z), coil_affine)
    profiles = siemens_basis(mesh_x, mesh_y, mesh_z)

    # Define coil1
    coil = Coil(profiles[..., :n_channel], coil_affine, constraints)
    return coil


def create_mask(ref):
    # Set up mask
    mask = shapes(ref, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)
    return mask


nz = 3  # Must be multiple of 3
a_unshimmed = create_unshimmed()
unshimmed_affine = create_unshimmed_affine()
coil_affine = unshimmed_affine * 2
coil_affine[3, 3] = 1
coil1 = create_coil(100, 100, nz, create_constraints(1000, -1000, 2000), unshimmed_affine)
affine = coil_affine * 0.75
affine[3, 3] = 1
coil2 = create_coil(150, 120, nz + 10, create_constraints(500, -500, 1500), affine)


@pytest.mark.parametrize(
    "unshimmed,un_affine,sph_coil,sph_coil2,mask", [(
        a_unshimmed,
        unshimmed_affine,
        coil1,
        coil2,
        create_mask(a_unshimmed)
    )]
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

    def test_shim_sequencer_slab_slices(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        """Test for slices arranged as a slab"""
        # Optimize
        z_slices = [(0, 1, 2)]
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil], mask, z_slices, method='least_squares')

        assert_results([sph_coil], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_dynamic_slices(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        """Test for slices arranged for dynamic shimming"""
        # Optimize
        z_slices = [(0,), (1,), (2,)]
        currents = shim_sequencer(unshimmed, un_affine, [sph_coil], mask, z_slices, method='least_squares')

        assert_results([sph_coil], unshimmed, un_affine, currents, mask, z_slices)

    def test_shim_sequencer_multi_slices(self, unshimmed, un_affine, sph_coil, sph_coil2, mask):
        """Test for slices arranged for multi slice"""
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

    if DEBUG:
        # Save fieldmap
        fname_fieldmap_2 = os.path.join(os.curdir, 'fig_fieldmap.nii.gz')
        nii_fieldmap = nib.Nifti1Image(unshimmed, un_affine)
        nib.save(nii_fieldmap, fname_fieldmap_2)

        # Save coil profiles as nifti
        fname_coil = os.path.join(os.curdir, 'fig_coil_orig.nii.gz')
        nii_coil = nib.Nifti1Image(coil[0].profile, coil[0].affine)
        nib.save(nii_coil, fname_coil)

        # save resampled coil profiles
        fname_coil_res = os.path.join(os.curdir, 'fig_coil_resampled.nii.gz')
        nii_coil = nib.Nifti1Image(opt.merged_coils, opt.unshimmed_affine)
        nib.save(nii_coil, fname_coil_res)

    correction_per_channel = np.zeros_like(opt.merged_coils)
    shimmed = np.zeros_like(unshimmed)
    for i_shim in range(len(z_slices)):
        correction_per_channel[..., z_slices[i_shim], :] = (currents[i_shim] * opt.merged_coils)[:, :, z_slices[i_shim], :]
        correction = np.sum(correction_per_channel[..., z_slices[i_shim], :], axis=3, keepdims=False)
        shimmed[..., z_slices[i_shim]] = unshimmed[..., z_slices[i_shim]] + correction

        sum_shimmed = np.sum(np.abs(mask[:, :, z_slices[i_shim]] * shimmed[:, :, z_slices[i_shim]]))
        sum_unshimmed = np.sum(np.abs(mask[:, :, z_slices[i_shim]] * unshimmed[:, :, z_slices[i_shim]]))
        print(f"\nshimmed: {sum_shimmed}, unshimmed: {sum_unshimmed}, current: \n{currents[i_shim, :]}")
        # assert sum_shimmed < sum_unshimmed

    if DEBUG:
        # save resampled coil profiles
        fname_correction = os.path.join(os.curdir, 'fig_correction.nii.gz')
        nii_correction = nib.Nifti1Image(correction_per_channel, opt.unshimmed_affine)
        nib.save(nii_correction, fname_correction)


def test_realtime_sequencer_phantom_data():
    """Test on the realtime sequencer using realtime zshimming data"""
    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')
    nii_fieldmap = nib.load(fname_fieldmap)
    unshimmed = nii_fieldmap.get_fdata()
    # Create affine closer to isocenter
    new_affine = np.eye(4)
    new_affine[:3, :3] = nii_fieldmap.affine[:3, :3]
    nii_fieldmap = nib.Nifti1Image(unshimmed, new_affine, header=nii_fieldmap.header)
    unshimmed = nii_fieldmap.get_fdata()

    # Set up mask
    # static
    nx, ny, nz, _ = nii_fieldmap.shape
    static_mask = shapes(unshimmed[..., 0], 'cube',
                         center_dim1=int(nx / 2) - 6,
                         center_dim2=int(ny / 2) - 5,
                         len_dim1=5, len_dim2=10, len_dim3=nz)

    # Riro
    riro_mask = shapes(unshimmed[..., 0], 'cube',
                       center_dim1=int(nx / 2) - 6,
                       center_dim2=int(ny / 2) - 5,
                       len_dim1=5, len_dim2=10, len_dim3=nz)

    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)

    # Path for json file
    fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_magnitude1.json')
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Calc pressure
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_data)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # Create Coil
    coil_affine = nii_fieldmap.affine
    coil = create_coil(150, 150, nz + 10, create_constraints(np.inf, -np.inf, np.inf), coil_affine)

    # Define the slices to shim with the proper convention
    slices = define_slices(unshimmed.shape[2], 1)

    # Find optimal currents
    currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_fieldmap, json_data, pmu, [coil],
                                                                                static_mask, riro_mask, slices,
                                                                                opt_method='least_squares')
    # currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_fieldmap, json_data, pmu, [coil],
    #                                                                             static_mask, riro_mask, slices,
    #                                                                             opt_method='pseudo_inverse')
    currents_riro_rms = currents_riro * p_rms

    print(f"\nSlices: {slices}"
          f"\nFieldmap affine:\n{nii_fieldmap.affine}\n"
          f"Coil affine:\n{coil_affine}\n"
          f"Static currents:\n{currents_static}\n"
          f"Riro currents * p_rms:\n{currents_riro_rms}\n")

    # Calculate theoretical shimmed map
    # shim
    opt = Optimizer([coil], unshimmed[..., 0], nii_fieldmap.affine)
    shimmed_static_riro = np.zeros_like(unshimmed)
    shimmed_static = np.zeros_like(unshimmed)
    shimmed_riro = np.zeros_like(unshimmed)
    masked_shim_static_riro = np.zeros_like(unshimmed)
    masked_shim_static = np.zeros_like(unshimmed)
    masked_shim_riro = np.zeros_like(unshimmed)
    masked_unshimmed = np.zeros_like(unshimmed)
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(currents_static[i_shim] *
                                   opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]

        # Calculate the riro coil profiles
        riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_static
            shimmed_static_riro[..., slices[i_shim], i_t] = shimmed_static[..., slices[i_shim], i_t] + correction_riro
            shimmed_riro[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_riro

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(static_mask[:, :, slices[i_shim]] * shimmed_static[:, :, slices[i_shim], i_t]))
            sum_shimmed_static_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_static_riro[:, :, slices[i_shim], i_t]))
            sum_shimmed_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_riro[:, :, slices[i_shim], i_t]))
            sum_unshimmed = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * unshimmed[:, :, slices[i_shim], i_t]))
            print(f"\ni_shim: {i_shim}, t: {i_t}"
                  f"\nshimmed: {sum_shimmed_static_riro}, unshimmed: {sum_unshimmed}, "
                  f"Static currents:\n{currents_static[i_shim]}\n"
                  f"Riro currents:\n{currents_riro[i_shim] * (acq_pressures[i_t] - mean_p)}\n")

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

            assert sum_shimmed_static_riro < sum_unshimmed

    if DEBUG:
        n_t = nii_fieldmap.shape[3]
        for i_t in range(n_t):
            # static mask and riro are the same
            masked_shim_static[..., i_t] = static_mask * shimmed_static[..., i_t]
            masked_shim_static_riro[..., i_t] = riro_mask * shimmed_static_riro[..., i_t]
            masked_shim_riro[..., i_t] = shimmed_riro[..., i_t]
            masked_unshimmed[..., i_t] = riro_mask * unshimmed[..., i_t]

        plot_shimmed_vs_unshimmed(0, 0, masked_shim_static_riro, masked_shim_static, masked_unshimmed,
                                  shimmed_static_riro, shimmed_static, unshimmed)

        # plot shimmed and unshimmed trace
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.plot(shim_trace_static_riro, label='shimmed static + riro')
        ax.plot(shim_trace_static, label='shimmed static')
        ax.plot(shim_trace_riro, label='shimmed_riro')
        ax.plot(unshimmed_trace, label='unshimmed')
        ax.legend()
        ax.set_ylim(0, max(unshimmed_trace))
        ax.set_title("Unshimmed vs shimmed values")
        fname_figure = os.path.join(os.curdir, 'fig_trace_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)

        plot_pressure_points(acq_pressures)
        save_nii(nii_fieldmap, coil, opt)
        print_metrics(static_mask, nii_fieldmap, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro)


def test_realtime_sequencer_fake_data():
    """Test on the realtime sequencer using simulated data"""

    # fake[..., 0] contains the original linear fieldmap. This repeats the linear fieldmap over the 3rd dim and scale
    # down
    fake = create_unshimmed()
    fake_temp = np.zeros([100, 100, 3, 4])
    lin = np.repeat(fake[:, :, 0, np.newaxis], 3, axis=2) / 10
    fake_temp[..., 0] = fake + lin
    fake_temp[..., 1] = fake
    fake_temp[..., 2] = fake - lin
    fake_temp[..., 3] = fake
    nii_fieldmap = nib.Nifti1Image(fake_temp, create_unshimmed_affine())
    unshimmed = nii_fieldmap.get_fdata()

    # Set up mask
    # static
    nx, ny, nz, _ = nii_fieldmap.shape
    static_mask = shapes(unshimmed[..., 0], 'cube',
                         center_dim1=int(nx / 2) - 6,
                         center_dim2=int(ny / 2) - 5,
                         len_dim1=20, len_dim2=20, len_dim3=nz)

    # Riro
    riro_mask = shapes(unshimmed[..., 0], 'cube',
                       center_dim1=int(nx / 2) - 6,
                       center_dim2=int(ny / 2) - 5,
                       len_dim1=20, len_dim2=20, len_dim3=nz)

    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)
    # Change pmu so that it uses fake data
    pmu.data = np.array([3000, 2000, 1000, 2000])
    pmu.stop_time_mdh = 750
    pmu.start_time_mdh = 0

    # Calc pressure
    json_data = {'RepetitionTime': 250 / 1000, 'AcquisitionTime': "00:00:00.000000"}
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_data)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # Create Coil
    coil_affine = nii_fieldmap.affine
    coil = create_coil(150, 150, nz + 10, create_constraints(np.inf, -np.inf, np.inf), coil_affine)

    # Define the slices to shim with the proper convention
    slices = define_slices(unshimmed.shape[2], 1)

    # Find optimal currents
    currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_fieldmap, json_data, pmu, [coil],
                                                                                static_mask, riro_mask, slices,
                                                                                opt_method='least_squares')
    # currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_fieldmap, json_data, pmu, [coil],
    #                                                                             static_mask, riro_mask, slices,
    #                                                                             opt_method='pseudo_inverse')
    currents_riro_rms = currents_riro * p_rms

    print(f"\nSlices: {slices}"
          f"\nFieldmap affine:\n{nii_fieldmap.affine}\n"
          f"Coil affine:\n{coil_affine}\n"
          f"Static currents:\n{currents_static}\n"
          f"Riro currents * p_rms:\n{currents_riro_rms}\n")

    # Calculate theoretical shimmed map
    # shim
    opt = Optimizer([coil], unshimmed[..., 0], nii_fieldmap.affine)
    shimmed_static_riro = np.zeros_like(unshimmed)
    shimmed_static = np.zeros_like(unshimmed)
    shimmed_riro = np.zeros_like(unshimmed)
    masked_shim_static_riro = np.zeros_like(unshimmed)
    masked_shim_static = np.zeros_like(unshimmed)
    masked_shim_riro = np.zeros_like(unshimmed)
    masked_unshimmed = np.zeros_like(unshimmed)
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(currents_static[i_shim] *
                                   opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]

        # Calculate the riro coil profiles
        riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_static
            shimmed_static_riro[..., slices[i_shim], i_t] = shimmed_static[..., slices[i_shim], i_t] + correction_riro
            shimmed_riro[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_riro

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(static_mask[:, :, slices[i_shim]] * shimmed_static[:, :, slices[i_shim], i_t]))
            sum_shimmed_static_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_static_riro[:, :, slices[i_shim], i_t]))
            sum_shimmed_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_riro[:, :, slices[i_shim], i_t]))
            sum_unshimmed = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * unshimmed[:, :, slices[i_shim], i_t]))
            print(f"\ni_shim: {i_shim}, t: {i_t}"
                  f"\nshimmed: {sum_shimmed_static_riro}, unshimmed: {sum_unshimmed}, "
                  f"Static currents:\n{currents_static[i_shim]}\n"
                  f"Riro currents:\n{currents_riro[i_shim] * (acq_pressures[i_t] - mean_p)}\n")

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

            assert sum_shimmed_static_riro < sum_unshimmed

    # reshape to slice x timepoint
    nt = nii_fieldmap.shape[3]
    n_slice = nii_fieldmap.shape[2]
    shim_trace_static = np.array(shim_trace_static).reshape(n_slice, nt)
    shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_slice, nt)
    shim_trace_riro = np.array(shim_trace_riro).reshape(n_slice, nt)
    unshimmed_trace = np.array(unshimmed_trace).reshape(n_slice, nt)

    if DEBUG:
        # Create mashed shim and unshimmed variables
        for i_t in range(nt):
            # static mask and riro are the same
            masked_shim_static[..., i_t] = static_mask * shimmed_static[..., i_t]
            masked_shim_static_riro[..., i_t] = riro_mask * shimmed_static_riro[..., i_t]
            masked_shim_riro[..., i_t] = shimmed_riro[..., i_t]
            masked_unshimmed[..., i_t] = riro_mask * unshimmed[..., i_t]

        # Plot figures of the shimmined and unshimmend maps
        plot_shimmed_vs_unshimmed(0, 0, masked_shim_static_riro, masked_shim_static, masked_unshimmed,
                                  shimmed_static_riro, shimmed_static, unshimmed)

        # plot shimmed and unshimmed trace
        fig = Figure(figsize=(10, 14))
        for i_slice in range(n_slice):
            ax = fig.add_subplot(n_slice, 1, i_slice + 1)
            ax.plot(shim_trace_static_riro[i_slice, :], label='shimmed static + riro')
            ax.plot(shim_trace_static[i_slice, :], label='shimmed static')
            ax.plot(shim_trace_riro[i_slice, :], label='shimmed_riro')
            ax.plot(unshimmed_trace[i_slice, :], label='unshimmed')
            ax.set_xlabel('Timepoints')
            ax.set_ylabel('Sum over the ROI')
            ax.legend()
            ax.set_ylim(0, max(unshimmed_trace[i_slice, :]))
            ax.set_title(f"Unshimmed vs shimmed values: slice {i_slice}")
        fname_figure = os.path.join(os.curdir, 'fig_trace_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)

        plot_pressure_points(acq_pressures)
        save_nii(nii_fieldmap, coil, opt)
        print_metrics(static_mask, nii_fieldmap, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro)

    # Todo:
    # Use same affine for coil and for defining siemens basis
    #
    # Look at input of tests if the coil profiles are at the same place as the unshimmed map


def test_sequencer_anat_res_before():
    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')
    nii_fieldmap = nib.load(fname_fieldmap)

    # anat image
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)

    # Set up mask
    # static
    nx, ny, nz = nii_anat.shape
    static_mask = shapes(nii_anat.get_fdata(), 'cube',
                         center_dim1=int(nx / 2),
                         center_dim2=int(ny / 2),
                         len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nii_anat.affine)

    # Riro
    riro_mask = shapes(nii_anat.get_fdata(), 'cube',
                       center_dim1=int(nx / 2),
                       center_dim2=int(ny / 2),
                       len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_riro = nib.Nifti1Image(riro_mask.astype(int), nii_anat.affine)

    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)

    # Path for json file
    fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_magnitude1.json')
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Create Coil
    coil_affine = nii_fieldmap.affine
    coil = create_coil(nii_anat.shape[0], nii_anat.shape[1], nii_anat.shape[2], create_constraints(np.inf, -np.inf, np.inf, n_channels=3), nii_anat.affine, n_channel=3)

    # Resample fieldmap on the anat
    nii_res_fieldmap = resample_from_to(nii_fieldmap, nii_anat, mode='nearest')

    # Define the slices to shim with the proper convention
    slices = define_slices(nii_anat.shape[2], 1)

    # Find optimal currents
    # currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_res_fieldmap, json_data, pmu, [coil],
    #                                                                             static_mask, riro_mask, slices,
    #                                                                             opt_method='least_squares')
    currents_static, currents_riro, mean_p, p_rms = shim_realtime_pmu_sequencer(nii_res_fieldmap, json_data, pmu, [coil],
                                                                                static_mask, riro_mask, slices,
                                                                                opt_method='pseudo_inverse')
    currents_riro_rms = currents_riro * p_rms

    print(f"\nSlices: {slices}"
          f"\nFieldmap affine:\n{nii_fieldmap.affine}\n"
          f"Coil affine:\n{coil_affine}\n"
          f"Static currents:\n{currents_static}\n"
          f"Riro currents:\n{currents_riro}\n"
          f"Riro currents * p_rms:\n{currents_riro_rms}\n")

    unshimmed = nii_res_fieldmap.get_fdata()
    # Calc pressure
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_data)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    opt = Optimizer([coil], nii_res_fieldmap.get_fdata()[..., 0], nii_anat.affine)
    shimmed_static_riro = np.zeros_like(unshimmed)
    shimmed_static = np.zeros_like(unshimmed)
    shimmed_riro = np.zeros_like(unshimmed)
    masked_shim_static_riro = np.zeros_like(unshimmed)
    masked_shim_static = np.zeros_like(unshimmed)
    masked_shim_riro = np.zeros_like(unshimmed)
    masked_unshimmed = np.zeros_like(unshimmed)
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(currents_static[i_shim] *
                                   opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]

        # Calculate the riro coil profiles
        riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)[..., slices[i_shim]]
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_static
            shimmed_static_riro[..., slices[i_shim], i_t] = shimmed_static[..., slices[i_shim], i_t] + correction_riro
            shimmed_riro[..., slices[i_shim], i_t] = unshimmed[..., slices[i_shim], i_t] + correction_riro

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(static_mask[:, :, slices[i_shim]] * shimmed_static[:, :, slices[i_shim], i_t]))
            sum_shimmed_static_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_static_riro[:, :, slices[i_shim], i_t]))
            sum_shimmed_riro = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * shimmed_riro[:, :, slices[i_shim], i_t]))
            sum_unshimmed = np.sum(np.abs(riro_mask[:, :, slices[i_shim]] * unshimmed[:, :, slices[i_shim], i_t]))
            print(f"\ni_shim: {i_shim}, t: {i_t}"
                  f"\nshimmed: {sum_shimmed_static_riro}, unshimmed: {sum_unshimmed}, "
                  f"Static currents:\n{currents_static[i_shim]}\n"
                  f"Riro currents:\n{currents_riro[i_shim] * (acq_pressures[i_t] - mean_p)}\n")

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

            assert sum_shimmed_static_riro < sum_unshimmed

    # reshape to slice x timepoint
    nt = nii_res_fieldmap.shape[3]
    n_shim = len(slices)
    shim_trace_static = np.array(shim_trace_static).reshape(n_shim, nt)
    shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_shim, nt)
    shim_trace_riro = np.array(shim_trace_riro).reshape(n_shim, nt)
    unshimmed_trace = np.array(unshimmed_trace).reshape(n_shim, nt)

    if DEBUG:
        # Create mashed shim and unshimmed variables
        for i_t in range(nt):
            # static mask and riro are the same
            masked_shim_static[..., i_t] = static_mask * shimmed_static[..., i_t]
            masked_shim_static_riro[..., i_t] = riro_mask * shimmed_static_riro[..., i_t]
            masked_shim_riro[..., i_t] = shimmed_riro[..., i_t]
            masked_unshimmed[..., i_t] = riro_mask * unshimmed[..., i_t]

        # Plot figures of the shimmined and unshimmend maps
        plot_shimmed_vs_unshimmed(0, 0, masked_shim_static_riro, masked_shim_static, masked_unshimmed,
                                  shimmed_static_riro, shimmed_static, unshimmed)

        plot_currents(currents_static, currents_riro_rms)

        # plot shimmed and unshimmed trace
        fig = Figure(figsize=(10, 50))
        for i_slice in range(n_shim):
            ax = fig.add_subplot(n_shim, 1, i_slice + 1)
            ax.plot(shim_trace_static_riro[i_slice, :], label='shimmed static + riro')
            ax.plot(shim_trace_static[i_slice, :], label='shimmed static')
            ax.plot(shim_trace_riro[i_slice, :], label='shimmed_riro')
            ax.plot(unshimmed_trace[i_slice, :], label='unshimmed')
            ax.set_xlabel('Timepoints')
            ax.set_ylabel('Sum over the ROI')
            ax.legend()
            ax.set_ylim(0, max(unshimmed_trace[i_slice, :]))
            ax.set_title(f"Unshimmed vs shimmed values: slice {i_slice}")
        fname_figure = os.path.join(os.curdir, 'fig_trace_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)

        plot_pressure_points(acq_pressures)
        save_nii(nii_res_fieldmap, coil, opt, nii_mask_static)
        print_metrics(static_mask, nii_res_fieldmap, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro)


def test_sequencer_anat_res_mask():
    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')
    nii_fieldmap = nib.load(fname_fieldmap)

    # anat image
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)

    # Set up mask
    # static
    nx, ny, nz = nii_anat.shape
    static_mask = shapes(nii_anat.get_fdata(), 'cube',
                         center_dim1=int(nx / 2),
                         center_dim2=int(ny / 2),
                         len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nii_anat.affine)

    # Riro
    # riro_mask = shapes(nii_anat.get_fdata(), 'cube',
    #                    center_dim1=int(nx / 2),
    #                    center_dim2=int(ny / 2),
    #                    len_dim1=5, len_dim2=5, len_dim3=nz)
    riro_mask = static_mask

    nii_mask_riro = nib.Nifti1Image(riro_mask.astype(int), nii_anat.affine)

    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)

    # Path for json file
    fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_magnitude1.json')
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Calc pressure
    acq_timestamps = get_acquisition_times(nii_fieldmap, json_data)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # Create Coil
    coil_affine = nii_fieldmap.affine
    coil = create_coil(150, 150, nz + 10, create_constraints(np.inf, -np.inf, np.inf), coil_affine)

    # Define the slices to shim with the proper convention
    slices = define_slices(nii_anat.shape[2], 5, method='sequential')

    # Find optimal currents
    output = shim_realtime_pmu_sequencer_anat(nii_fieldmap, json_data, nii_anat, pmu, [coil], nii_mask_static,
                                              nii_mask_riro, slices, opt_method='least_squares')
    currents_static, currents_riro, mean_p, p_rms = output

    # Scale according to rms
    currents_riro_rms = currents_riro * p_rms

    # Print some outputs
    print(f"\nSlices: {slices}"
          f"\nFieldmap affine:\n{nii_fieldmap.affine}\n"
          f"Coil affine:\n{coil_affine}\n"
          f"Static currents:\n{currents_static}\n"
          f"Riro currents * p_rms:\n{currents_riro_rms}\n")

    # Calculate theoretical shimmed map
    # shim
    unshimmed = nii_fieldmap.get_fdata()
    nii_target = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine)
    opt = Optimizer([coil], unshimmed[..., 0], nii_fieldmap.affine)
    shape = unshimmed.shape + (len(slices),)
    shimmed_static_riro = np.zeros(shape)
    shimmed_static = np.zeros(shape)
    shimmed_riro = np.zeros(shape)
    masked_shim_static_riro = np.zeros(shape)
    masked_shim_static = np.zeros(shape)
    masked_shim_riro = np.zeros(shape)
    masked_unshimmed = np.zeros(shape)
    masked_fieldmap = np.zeros(unshimmed[..., 0].shape + (len(slices),))
    shim_trace_static_riro = []
    shim_trace_static = []
    shim_trace_riro = []
    unshimmed_trace = []
    for i_shim in range(len(slices)):
        # Calculate static correction
        correction_static = np.sum(currents_static[i_shim] *
                                   opt.merged_coils, axis=3, keepdims=False)

        # Calculate the riro coil profiles
        riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        masked_fieldmap[..., i_shim] = resample_mask(nii_mask_static, nii_target, slices[i_shim]).get_fdata()
        for i_t in range(nii_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
            shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
            shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

            # Calculate masked shim
            masked_shim_static[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static[..., i_t, i_shim]
            masked_shim_static_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static_riro[..., i_t, i_shim]
            masked_shim_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_riro[..., i_t, i_shim]
            masked_unshimmed[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * unshimmed[..., i_t]

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(masked_shim_static[..., i_t, i_shim]))
            sum_shimmed_static_riro = np.sum(np.abs(masked_shim_static_riro[..., i_t, i_shim]))
            sum_shimmed_riro = np.sum(np.abs(masked_shim_riro[..., i_t, i_shim]))
            sum_unshimmed = np.sum(np.abs(masked_unshimmed[..., i_t, i_shim]))
            print(f"\ni_shim: {i_shim}, t: {i_t}"
                  f"\nshimmed static: {sum_shimmed_static}, shimmed static+riro: {sum_shimmed_static_riro}, "
                  f"unshimmed: {sum_unshimmed}\n"
                  f"Static currents:\n{currents_static[i_shim]}\n"
                  f"Riro currents:\n{currents_riro[i_shim] * (acq_pressures[i_t] - mean_p)}\n")

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

            assert sum_shimmed_static_riro < sum_unshimmed

    if DEBUG:

        # reshape to slice x timepoint
        nt = unshimmed.shape[3]
        n_shim = len(slices)
        shim_trace_static = np.array(shim_trace_static).reshape(n_shim, nt)
        shim_trace_static_riro = np.array(shim_trace_static_riro).reshape(n_shim, nt)
        shim_trace_riro = np.array(shim_trace_riro).reshape(n_shim, nt)
        unshimmed_trace = np.array(unshimmed_trace).reshape(n_shim, nt)

        # Plot Static and RIRO
        i_t = 0
        i_slice = 0
        i_shim = 0

        min_value = min(masked_shim_static_riro[..., i_slice, i_t, i_shim].min(),
                        masked_shim_static[..., i_slice, i_t, i_shim].min(),
                        masked_unshimmed[..., i_slice, i_t, i_shim].min())
        max_value = max(masked_shim_static_riro[..., i_slice, i_t, i_shim].max(),
                        masked_shim_static[..., i_slice, i_t, i_shim].max(),
                        masked_unshimmed[..., i_slice, i_t, i_shim].max())

        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 3, 1)
        im = ax.imshow(np.rot90(masked_shim_static_riro[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
        fig.colorbar(im)
        ax.set_title("masked_shim static + riro")
        ax = fig.add_subplot(2, 3, 2)
        im = ax.imshow(np.rot90(masked_shim_static[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
        fig.colorbar(im)
        ax.set_title("masked_shim static")
        ax = fig.add_subplot(2, 3, 3)
        im = ax.imshow(np.rot90(masked_unshimmed[..., i_slice, i_t, i_shim]), vmin=min_value, vmax=max_value)
        fig.colorbar(im)
        ax.set_title("masked_unshimmed")

        ax = fig.add_subplot(2, 3, 4)
        im = ax.imshow(np.rot90(shimmed_static_riro[..., i_slice, i_t, i_shim]))
        fig.colorbar(im)
        ax.set_title("shim static + riro")
        ax = fig.add_subplot(2, 3, 5)
        im = ax.imshow(np.rot90(shimmed_static[..., i_slice, i_t, i_shim]))
        fig.colorbar(im)
        ax.set_title("shim static")
        ax = fig.add_subplot(2, 3, 6)
        im = ax.imshow(np.rot90(unshimmed[..., i_slice, i_t]))
        fig.colorbar(im)
        ax.set_title("unshimmed")
        fname_figure = os.path.join(os.curdir, 'fig_realtime_masked_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)

        plot_currents(currents_static, currents_riro_rms)

        # plot shimmed and unshimmed trace
        fig = Figure(figsize=(10, 50))
        for i_shim in range(n_shim):
            ax = fig.add_subplot(n_shim, 1, i_shim + 1)
            ax.plot(shim_trace_static_riro[i_shim, :], label='shimmed static + riro')
            ax.plot(shim_trace_static[i_shim, :], label='shimmed static')
            ax.plot(shim_trace_riro[i_shim, :], label='shimmed_riro')
            ax.plot(unshimmed_trace[i_shim, :], label='unshimmed')
            ax.set_xlabel('Timepoints')
            ax.set_ylabel('Sum over the ROI')
            ax.legend()
            ax.set_ylim(0, max(unshimmed_trace[i_slice, :]))
            ax.set_title(f"Unshimmed vs shimmed values: slice {i_shim}")
        fname_figure = os.path.join(os.curdir, 'fig_trace_shimmed_vs_unshimmed.png')
        fig.savefig(fname_figure)

        plot_pressure_points(acq_pressures)
        save_nii(nii_fieldmap, coil, opt, nii_mask_static)
        # print_metrics(static_mask, nii_fieldmap, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro)

        # Problem: When slice is first or last, it removes it from the ROI entirely if mode is cst, if nearest, it
        # includes a bunch of unwanted pixels outside the roi


def plot_currents(static, riro):
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(static[:, 0], label='Static dim0 currents through shims')
    ax.plot(static[:, 1], label='Static dim1 currents through shims')
    ax.plot(static[:, 2], label='Static dim2 currents through shims')
    ax.plot(riro[:, 0], label='Riro dim0 currents through shims')
    ax.plot(riro[:, 1], label='Riro dim1 currents through shims')
    ax.plot(riro[:, 2], label='Riro dim2 currents through shims')
    ax.set_xlabel('i_shims')
    ax.set_ylabel('Currrents')
    ax.legend()
    ax.set_title("Static and riro currents through shims")
    fname_figure = os.path.join(os.curdir, 'fig_currents.png')
    fig.savefig(fname_figure)


def plot_pressure_points(acq_pressures):
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(acq_pressures, label='pressures')
    ax.legend()
    ax.set_ylim(0, 4095)
    ax.set_title("Pressures vs time points")
    fname_figure = os.path.join(os.curdir, 'fig_trace_pressures.png')
    fig.savefig(fname_figure)


def plot_shimmed_vs_unshimmed(timepoint, slice, masked_shim_static_riro, masked_shim_static, masked_unshimmed,
                              shimmed_static_riro, shimmed_static, unshimmed):
    # Plot Static and RIRO
    i_t = timepoint
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 3, 1)
    im = ax.imshow(np.rot90(masked_shim_static_riro[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("masked_shim static + riro")
    ax = fig.add_subplot(2, 3, 2)
    im = ax.imshow(np.rot90(masked_shim_static[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("masked_shim static")
    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(np.rot90(masked_unshimmed[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("masked_unshimmed")

    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(np.rot90(shimmed_static_riro[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("shim static + riro")
    ax = fig.add_subplot(2, 3, 5)
    im = ax.imshow(np.rot90(shimmed_static[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("shim static")
    ax = fig.add_subplot(2, 3, 6)
    im = ax.imshow(np.rot90(unshimmed[..., slice, i_t]))
    fig.colorbar(im)
    ax.set_title("unshimmed")
    fname_figure = os.path.join(os.curdir, 'fig_realtime_masked_shimmed_vs_unshimmed.png')
    fig.savefig(fname_figure)


def save_nii(nii_fieldmap, coil, opt, nii_mask):
    # save mask
    fname_mask = os.path.join(os.curdir, 'fig_mask.nii.gz')
    nib.save(nii_mask, fname_mask)

    # Save fieldmap
    fname_fieldmap_2 = os.path.join(os.curdir, 'fig_fieldmap.nii.gz')
    nib.save(nii_fieldmap, fname_fieldmap_2)

    # Save coil profiles as nifti
    fname_coil = os.path.join(os.curdir, 'fig_coil_orig.nii.gz')
    nii_coil = nib.Nifti1Image(coil.profile, coil.affine)
    nib.save(nii_coil, fname_coil)

    # save resampled coil profiles
    fname_coil_res = os.path.join(os.curdir, 'fig_coil_resampled.nii.gz')
    nii_coil = nib.Nifti1Image(opt.merged_coils, opt.unshimmed_affine)
    nib.save(nii_coil, fname_coil_res)


def print_metrics(static_mask, nii_fieldmap, unshimmed, shimmed_static, shimmed_static_riro, shimmed_riro):
    # metric to isolate temporal and static component
    # Temporal: Compute the STD across time pixelwise, and then compute the mean across pixels.
    # Static: Compute the MEAN across time pixelwise, and then compute the STD across pixels.

    rep_mask = np.repeat(static_mask[..., np.newaxis], nii_fieldmap.shape[3], 3)
    ma_unshimmed = np.ma.array(unshimmed, mask=rep_mask == False)
    ma_shim_static = np.ma.array(shimmed_static, mask=rep_mask == False)
    ma_shim_static_riro = np.ma.array(shimmed_static_riro, mask=rep_mask == False)
    ma_shim_riro = np.ma.array(shimmed_riro, mask=rep_mask == False)

    # Temporal
    temp_shim_static = np.ma.mean(np.ma.std(ma_shim_static, 3))
    temp_shim_static_riro = np.ma.mean(np.ma.std(ma_shim_static_riro, 3))
    temp_shim_riro = np.ma.mean(np.ma.std(ma_shim_riro, 3))
    temp_unshimmed = np.ma.mean(np.ma.std(ma_unshimmed, 3))

    # Static
    static_shim_static = np.ma.std(np.ma.mean(ma_shim_static, 3))
    static_shim_static_riro = np.ma.std(np.ma.mean(ma_shim_static_riro, 3))
    static_shim_riro = np.ma.std(np.ma.mean(ma_shim_riro, 3))
    static_unshimmed = np.ma.std(np.ma.mean(ma_unshimmed, 3))
    print(f"\nTemporal: Compute the STD across time pixelwise, and then compute the mean across pixels."
          f"\ntemp_shim_static: {temp_shim_static}"
          f"\ntemp_shim_static_riro: {temp_shim_static_riro}"
          f"\ntemp_shim_riro: {temp_shim_riro}"
          f"\ntemp_unshimmed: {temp_unshimmed}"
          f"\nStatic: Compute the MEAN across time pixelwise, and then compute the STD across pixels."
          f"\nstatic_shim_static: {static_shim_static}"
          f"\nstatic_shim_static_riro: {static_shim_static_riro}"
          f"\nstatic_shim_riro: {static_shim_riro}"
          f"\nstatic_unshimmed: {static_unshimmed}")


from shimmingtoolbox.shim.sequencer import resample_mask
def test_resample_mask():

    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')
    nii_fieldmap = nib.load(fname_fieldmap)
    nii_target = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine)

    # anat image
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)

    # Set up mask
    # static
    nx, ny, nz = nii_anat.shape
    static_mask = shapes(nii_anat.get_fdata(), 'cube',
                         center_dim1=int(nx / 2),
                         center_dim2=int(ny / 2),
                         len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nii_anat.affine)

    nii_mask_res = resample_mask(nii_mask_static, nii_target, (2,))
    nib.save(nii_mask_res, os.path.join(os.curdir, "fig_res_mask.nii.gz"))

    nib.save(nii_mask_static, os.path.join(os.curdir, "fig_full_mask.nii.gz"))
