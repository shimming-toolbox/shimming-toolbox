#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import logging
import nibabel as nib
import numpy as np
import os
import pytest
import tempfile
import pathlib
import copy

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.optimizer.basic_optimizer import Optimizer
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.shim.sequencer import ShimSequencer, RealTimeSequencer, resample_mask
from shimmingtoolbox.shim.sequencer import define_slices, extend_slice, parse_slices
from shimmingtoolbox.shim.sequencer import shim_max_intensity
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.utils import set_all_loggers
from shimmingtoolbox.files.NiftiFieldMap import NiftiFieldMap
from shimmingtoolbox.files.NiftiTarget import NiftiTarget
from shimmingtoolbox.files.NiftiMask import NiftiMask

logger = logging.getLogger(__name__)
set_all_loggers('info')
DEBUG = False


def create_fieldmap(n_slices=3, is_realtime=False):
    # Set up 2-dimensional unshimmed fieldmaps
    num_vox = 100
    model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
    model_obj.generate_deltaB0('z', [0.025, 2])
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
    unshimmed = np.zeros([num_vox, num_vox, n_slices])
    for i_n in range(n_slices // 3):
        unshimmed[:, :, 3 * i_n] = b0_map
        unshimmed[:, :, (3 * i_n) + 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
        unshimmed[:, :, (3 * i_n) + 2] = unshimmed[:, :, 0] ** 2

    if is_realtime:
        # For realtime, we need a 4D fieldmap
        unshimmed = np.repeat(unshimmed[:, :, :, np.newaxis], 4, axis=3)

    nii_fmap = nib.Nifti1Image(unshimmed, create_unshimmed_affine())

    # Save in tmp directory
    nib.save(nii_fmap, os.path.join(__dir_testing__, 'fieldmap.nii.gz'))
    # save a fake json file
    json_fmap = {'SliceThickness': 3}

    # Load the fieldmap
    nif_fmap = NiftiFieldMap(os.path.join(__dir_testing__, 'fieldmap.nii.gz'),
                             dilation_kernel_size=3, json=json_fmap, is_realtime=is_realtime)

    return nif_fmap


def create_unshimmed_affine():
    # return np.array([[0., 0., 3., 1],
    #                  [-2.91667008, 0., 0., 2],
    #                  [0., 2.91667008, 0., 3],
    #                  [0., 0., 0., 1.]])
    return np.eye(4)


def create_constraints(max_coef, min_coef, sum_coef, n_channels=8):
    # Set up bounds for output currents
    bounds = {"coil": []}
    for _ in range(n_channels):
        bounds['coil'].append((min_coef, max_coef))

    constraints = {
        "name": "test",
        "coef_sum_max": sum_coef,
        "coef_channel_minmax": bounds
    }
    return constraints


def create_coil(n_x, n_y, n_z, constraints, coil_affine, n_channel=8):
    # Set up spherical harmonics coil profile
    mesh_x, mesh_y, mesh_z = generate_meshgrid((n_x, n_y, n_z), coil_affine)
    profiles = siemens_basis(mesh_x, mesh_y, mesh_z)

    # Define coil
    coil = Coil(profiles[..., :n_channel], coil_affine, constraints)
    return coil

nz = 3
nif_to_shim = create_fieldmap()

# Create coil profiles
unshimmed_affine = create_unshimmed_affine()
coil_affine = unshimmed_affine * 2
coil_affine[3, 3] = 1
# Coil with same #of pixel and same affine as fieldmap
coil1 = create_coil(100, 100, nz, create_constraints(1000, -1000, 2000), unshimmed_affine)
affine = coil_affine * 0.75
affine[3, 3] = 1
# Coil with different affine and different # of pixels
coil2 = create_coil(150, 120, nz + 10, create_constraints(500, -500, 1500), affine)

# Create target
target = np.ones((50, 50, 3))
nii_target = nib.Nifti1Image(target, affine=affine)
json_target = {'SliceThickness': 3}
# Save in tmp directory

nib.save(nii_target, os.path.join(__dir_testing__, 'target.nii.gz'))
# Load the target image
nif_target = NiftiTarget(os.path.join(__dir_testing__, 'target.nii.gz'), json=json_target)

# Create mask
static_mask = shapes(target, 'cube', len_dim1=10, len_dim2=10, len_dim3=nz)
nii_mask = nib.Nifti1Image(static_mask.astype(int), nii_target.affine, header=nii_target.header)
# Save in tmp directory
nib.save(nii_mask, os.path.join(__dir_testing__, 'mask.nii.gz'))
# Load the mask
nif_mask = NiftiMask(os.path.join(__dir_testing__, 'mask.nii.gz'))

@pytest.mark.parametrize(
    "nif_fieldmap,nif_target,nif_mask,sph_coil,sph_coil2", [(
            nif_to_shim,
            nif_target,
            nif_mask,
            coil1,
            coil2,
    )]
)
class TestSequencer(object):
    """Tests for shim_sequencer"""

    def test_shim_sequencer_lsq(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='least_squares')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_quad_prog(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='quad_prog')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_pseudo(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='pseudo_inverse')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_bfgs(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='bfgs')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_bfgs_ps_huber(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='bfgs', opt_criteria='ps_huber')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_mae(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='least_squares', opt_criteria='mae')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_ps_huber_lsq(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                                       method='least_squares', opt_criteria='ps_huber')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_2_coils_lsq(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices,
                                       [sph_coil, sph_coil2], method='least_squares')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil, sph_coil2], currents, slices)

    def test_shim_sequencer_2_coils_quad_prog(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices,
                                       [sph_coil, sph_coil2], method='quad_prog')
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil, sph_coil2], currents, slices)

    def test_shim_sequencer_coefs_are_none(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Coil with None constraints
        coil = create_coil(5, 5, nz, create_constraints(None, None, None), affine)
        # Optimize
        slices = define_slices(nif_target.shape[2], 1)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [coil], currents, slices)

    def test_shim_sequencer_slab_slices(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        """Test for slices arranged as a slab"""
        # Optimize
        slices = define_slices(nif_target.shape[2], nif_target.shape[2])
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_dynamic_slices(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        """Test for slices arranged for dynamic shimming"""
        # Optimize
        slices = [(0,), (1,), (2,)]
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_multi_slices(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        """Test for slices arranged for multi slice"""
        # Optimize
        slices = [(0, 2), (1,)]
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_wrong_optimizer(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = [(0, 2), (1,)]
        method = 'abc'
        with pytest.raises(KeyError, match=f"Method: {method} is not part of the supported optimizers"):
            ShimSequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil],
                          method=method).shim()

    # def test_shim_sequencer_wrong_units(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2, caplog):
    #     # Change the name of the units
    #     sph_coil2.units = "T"
    #     slices = [(0, 2), (1,)]
    #     shim_sequencer(nif_fieldmap, nif_target, nif_mask, slices, [sph_coil, sph_coil2])
    #     assert "The coils don't have matching units:" in caplog.text

    def test_shim_sequencer_diff_mask_affine(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = [(0, 2), (1,)]
        diff_affine = nif_mask.affine
        diff_affine[0, 0] = 2
        nii_diff_mask = nib.Nifti1Image(nif_mask.data, diff_affine, header=nif_mask.header)
        nif_diff_mask = nif_mask
        nif_diff_mask.set_nii(nii_diff_mask, nif_target)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_diff_mask, slices,
                                       [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_diff_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_diff_mask_shape(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        # Optimize
        slices = [(0, 2), (1,)]
        nii_diff_mask = nib.Nifti1Image(nif_mask.data[5:, ...], nif_mask.affine, header=nif_mask.header)
        nif_diff_mask = nif_mask
        nif_diff_mask.set_nii(nii_diff_mask, nif_target)
        sequencer_test = ShimSequencer(nif_fieldmap, nif_target, nif_diff_mask, slices,
                                       [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_diff_mask, [sph_coil], currents, slices)

    def test_shim_sequencer_4dmask_4d_target(self, nif_fieldmap, nif_target, nif_mask, sph_coil, sph_coil2):
        target_4d = np.repeat(nif_target.data[..., np.newaxis], 3, -1)
        mask_4d = np.repeat(nif_mask.data[..., np.newaxis], 2, -1)
        nii_4d_target = nib.Nifti1Image(target_4d, nif_target.affine, header=nif_target.header)
        nif_4d_target = nif_target
        nif_4d_target.set_nii(nii_4d_target)
        nii_4d_mask = nib.Nifti1Image(mask_4d, nif_mask.affine, header=nif_mask.header)
        nif_4d_mask = nif_mask
        nif_4d_mask.set_nii(nii_4d_mask, nif_target)

        slices = [(0, 2), (1,)]
        sequencer_test = ShimSequencer(nif_fieldmap, nif_4d_target, nif_4d_mask, slices,
                                       [sph_coil])
        currents = sequencer_test.shim()
        sequencer_test.eval(currents)
        assert_results(nif_fieldmap, nif_target, nif_mask, [sph_coil], currents, slices)


def assert_results(nif_fieldmap, nif_target, nif_mask, coil, currents, slices):
    # Calculate theoretical shimmed map
    unshimmed = nif_fieldmap.data
    opt = Optimizer(coil, unshimmed, nif_fieldmap.affine)

    if DEBUG:
        # Save fieldmap
        fname_fieldmap_2 = os.path.join(os.curdir, 'fig_fieldmap.nii.gz')
        nib.save(nif_fieldmap, fname_fieldmap_2)

        # Save target
        fname_target = os.path.join(os.curdir, 'fig_target.nii.gz')
        nib.save(nif_target, fname_target)

        # Save target mask
        fname_mask = os.path.join(os.curdir, 'fig_target_mask.nii.gz')
        nib.save(nif_mask, fname_mask)

        # Save coil profiles as nifti
        fname_coil = os.path.join(os.curdir, 'fig_coil_orig.nii.gz')
        nii_coil = nib.Nifti1Image(coil[0].profile, coil[0].affine)
        nib.save(nii_coil, fname_coil)

        # save resampled coil profiles
        fname_coil_res = os.path.join(os.curdir, 'fig_coil_resampled.nii.gz')
        nii_coil = nib.Nifti1Image(opt.merged_coils, opt.unshimmed_affine, header=nif_fieldmap.header)
        nib.save(nii_coil, fname_coil_res)

    correction_per_channel = np.zeros(opt.merged_coils.shape + (len(slices),))
    shimmed = np.zeros(unshimmed.shape + (len(slices),))
    mask_fieldmap = np.zeros(unshimmed.shape + (len(slices),))
    for i_shim in range(len(slices)):
        correction_per_channel[..., i_shim] = currents[i_shim] * opt.merged_coils
        correction = np.sum(correction_per_channel[..., i_shim], axis=3, keepdims=False)
        shimmed[..., i_shim] = unshimmed + correction

        mask_fieldmap[..., i_shim] = resample_mask(nif_mask.nii, nif_fieldmap.nii, slices[i_shim]).get_fdata()

        sum_shimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * shimmed[..., i_shim]))
        sum_unshimmed = np.sum(np.abs(mask_fieldmap[..., i_shim] * unshimmed))

        assert sum_shimmed <= sum_unshimmed


def define_rt_sim_inputs():
    # target image
    fname_target = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
    nif_target = NiftiTarget(fname_target)

    # fake[..., 0] contains the original linear fieldmap. This repeats the linear fieldmap over the 3rd dim and scale
    # down
    nz = 3
    nif_fieldmap = create_fieldmap(n_slices=nz, is_realtime=True)
    fake_temp = np.zeros([100, 100, nz, 4])
    lin = nif_fieldmap.data[..., 0] / 10
    fake_temp[..., 0] = nif_fieldmap.data[..., 0] + lin
    fake_temp[..., 1] = nif_fieldmap.data[..., 0]
    fake_temp[..., 2] = nif_fieldmap.data[..., 0] - lin
    fake_temp[..., 3] = nif_fieldmap.data[..., 0]
    fake_affine = nif_target.affine * 0.75
    fake_affine[:, 3] = nif_target.affine[:, 3]
    fake_affine[3, 3] = 1
    nii_fieldmap = nib.Nifti1Image(fake_temp, fake_affine)
    nif_fieldmap.set_nii(nii_fieldmap)

    # Set up mask
    # static
    nx, ny, nz = nif_target.shape
    static_mask = shapes(nif_target.data, 'cube', len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nif_target.affine, header=nif_target.header)
    # Save mask
    nib.save(nii_mask_static, os.path.join(__dir_testing__, 'mask_static.nii.gz'))
    # Load the mask
    nif_mask_static = NiftiMask(os.path.join(__dir_testing__, 'mask_static.nii.gz'), nif_target)

    nif_mask_riro = copy.deepcopy(nif_mask_static)

    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                              'sub-realtime_PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)
    # Change pmu so that it uses fake data. The fake data is essentially a sinusoid with 4 points
    fake_data = np.array([3000, 2000, 1000, 2000, 3000])
    pmu.set_data(fake_data)
    pmu.set_start_and_stop_times(0, 1000)

    # Define a dummy json data with the bare minimum fields and calculate the pressures
    json_data = {'RepetitionTime': 250 / 1000, 'AcquisitionTime': "00:00:00.000000"}
    nif_fieldmap.json = json_data
    # acq_timestamps = get_acquisition_times(nii_fieldmap, json_data, when='volume-start')

    # Create Coil
    coil_affine = nif_fieldmap.affine
    coil = create_coil(150, 150, nz + 10, create_constraints(np.inf, -np.inf, np.inf, n_channels=3),
                       coil_affine, n_channel=3)

    # Define the slices to shim with the proper convention
    slices = define_slices(nif_target.shape[2], 1, method='ascending')

    return nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro, slices, pmu, coil


nif_rt_fieldmap, nif_rt_target, nif_mask_rt_static, nif_mask_rt_riro, slices_rt, pmu_rt, coil_rt = \
    define_rt_sim_inputs()


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
class TestShimRTpmuSimData(object):
    """Tests for realtime Sequencer with simulated data"""

    def test_shim_realtime_pmu_sequencer_fake_data(self, nif_fieldmap, nif_target, nif_mask_static,
                                                   nif_mask_riro, slices, pmu, coil):
        """Test on the shim_realtime_pmu_sequencer using simulated data"""

        # Find optimal currents
        sequencer_realtime_test = RealTimeSequencer(nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro,
                                                    slices, pmu, [coil], [coil],
                                                    mask_dilation_kernel='sphere')
        output = sequencer_realtime_test.shim()
        currents_static, currents_riro, mean_p, p_rms = output
        sequencer_realtime_test.eval(currents_static, currents_riro, mean_p, p_rms)
        currents_riro_rms = currents_riro * p_rms

        print(f"\nSlices: {slices}"
              f"\nFieldmap affine:\n{nif_fieldmap.affine}\n"
              f"Coil affine:\n{coil.affine}\n"
              f"Static currents:\n{currents_static}\n"
              f"Riro currents * p_rms:\n{currents_riro_rms}\n")

        # Calculate theoretical shimmed map
        # shim
        unshimmed = nif_fieldmap.data
        nii_target = nib.Nifti1Image(nif_fieldmap.data[..., 0], nif_fieldmap.affine, header=nif_fieldmap.header)
        opt = Optimizer([coil], unshimmed[..., 0], nif_fieldmap.affine)
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
        data = pmu.get_data()

        for i_shim in range(len(slices)):
            # Calculate static correction
            correction_static = np.sum(currents_static[i_shim] * opt.merged_coils, axis=3, keepdims=False)

            # Calculate the riro coil profiles
            riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)

            masked_fieldmap[..., i_shim] = resample_mask(nif_mask_static.nii, nii_target, slices[i_shim],
                                                         dilation_kernel='sphere').get_fdata()
            for i_t in range(nif_fieldmap.shape[3]):
                # Apply the static and riro correction
                correction_riro = riro_profile * (data[i_t] - mean_p)
                shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
                shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
                shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

                # Calculate masked shim
                masked_shim_static[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static[..., i_t, i_shim]
                masked_shim_static_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static_riro[
                    ..., i_t, i_shim]
                masked_shim_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_riro[..., i_t, i_shim]
                masked_unshimmed[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * unshimmed[..., i_t]

                # Calculate the sum over the ROI
                sum_shimmed_static = np.sum(np.abs(masked_shim_static[..., i_t, i_shim]))
                sum_shimmed_static_riro = np.sum(np.abs(masked_shim_static_riro[..., i_t, i_shim]))
                sum_shimmed_riro = np.sum(np.abs(masked_shim_riro[..., i_t, i_shim]))
                sum_unshimmed = np.sum(np.abs(masked_unshimmed[..., i_t, i_shim]))

                # Create a 1D list of the sum of the shimmed and unshimmed maps
                shim_trace_static.append(sum_shimmed_static)
                shim_trace_static_riro.append(sum_shimmed_static_riro)
                shim_trace_riro.append(sum_shimmed_riro)
                unshimmed_trace.append(sum_unshimmed)

                assert sum_shimmed_static_riro <= sum_unshimmed

    def test_shim_sequencer_rt_larger_coil(self, nif_fieldmap, nif_target, nif_mask_static,
                                           nif_mask_riro, slices, pmu, coil):

        constraints = create_constraints(1000, -1000, 2000, 3)
        new_coil = create_coil(nif_fieldmap.shape[0], nif_fieldmap.shape[1], 3, constraints, nif_fieldmap.affine, 3)

        # Find optimal currents
        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro,
                                   slices, pmu, [new_coil], [new_coil]).shim()
        currents_static, currents_riro, mean_p, p_rms = output

        print(f"\nSlices: {slices}"
              f"\nFieldmap affine:\n{nif_fieldmap.affine}\n"
              f"Coil affine:\n{new_coil.affine}\n"
              f"Static currents:\n{currents_static}\n"
              f"Riro currents * p_rms:\n{currents_riro * p_rms}\n")

        assert np.all(currents_static.shape == (20, 3))

    def test_shim_sequencer_rt_bfgs(self, nif_fieldmap, nif_target, nif_mask_static,
                                           nif_mask_riro, slices, pmu, coil):
        # Optimize
        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro,
                                   slices, pmu, [coil], [coil], method='bfgs').shim()

        assert output[0].shape == (20, 3)

    def test_shim_sequencer_rt_kernel_line(self, nif_fieldmap, nif_target, nif_mask_static,
                                           nif_mask_riro, slices, pmu, coil):
        # Optimize
        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_mask_static, nif_mask_riro,
                                   slices, pmu, [coil], [coil], mask_dilation_kernel='line').shim()

        assert output[0].shape == (20, 3)

    def test_shim_sequencer_rt_diff_mask_shape_static(self, nif_fieldmap, nif_target, nif_mask_static,
                                                      nif_mask_riro, slices, pmu, coil):
        # Optimize
        nii_diff_mask = nib.Nifti1Image(nif_mask_static.data[5:, ...], nif_mask_static.affine,
                                        header=nif_mask_static.header)
        nif_diff_mask = nif_mask_static
        nif_diff_mask.set_nii(nii_diff_mask, nif_target)
        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_diff_mask, nif_mask_riro,
                                   slices, pmu, [coil], [coil]).shim()
        assert output[0].shape == (20, 3)

    def test_shim_sequencer_rt_diff_mask_shape_riro(self, nif_fieldmap, nif_target, nif_mask_static,
                                                    nif_mask_riro, slices, pmu, coil):
        # Optimize
        nii_diff_mask = nib.Nifti1Image(nif_mask_riro.data[5:, ...], nif_mask_riro.affine,
                                        header=nif_mask_riro.header)
        nif_diff_mask = nif_mask_riro
        nif_diff_mask.set_nii(nii_diff_mask, nif_target)

        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_mask_static, nif_diff_mask,
                                   slices, pmu, [coil], [coil]).shim()
        assert output[0].shape == (20, 3)

    def test_shim_sequencer_rt_diff_mask_affine(self, nif_fieldmap, nif_target, nif_mask_static,
                                                nif_mask_riro, slices, pmu, coil):
        # Optimize
        diff_affine = nif_mask_static.affine
        diff_affine[0, 0] = 2
        nii_diff_mask = nib.Nifti1Image(nif_mask_static.data, diff_affine, header=nif_mask_static.header)
        nif_diff_mask = nif_mask_static
        nif_diff_mask.set_nii(nii_diff_mask, nif_target)
        output = RealTimeSequencer(nif_fieldmap, nif_target, nif_diff_mask, nif_mask_riro,
                                   slices, pmu, [coil], [coil]).shim()
        assert output[0].shape == (20, 3)


def test_shim_realtime_pmu_sequencer_rt_zshim_data():
    """Tests for realtime Sequencer with real data"""
    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
    nif_fieldmap = NiftiFieldMap(fname_fieldmap, dilation_kernel_size=3, is_realtime=True)

    # target image
    fname_target = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
    nif_target = NiftiTarget(fname_target)

    # Set up mask
    # static
    nx, ny, nz = nif_target.shape
    static_mask = shapes(nif_target.data, 'cube', len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask = nib.Nifti1Image(static_mask.astype(int), nif_target.affine, header=nif_target.header)

    # save mask
    nib.save(nii_mask, os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_mask_static.nii.gz'))

    nif_static_mask = NiftiMask(os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_mask_static.nii.gz'), nif_target)
    nif_riro_mask = copy.deepcopy(nif_static_mask)
    # Pmu
    fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                              'sub-realtime_PMUresp_signal.resp')
    pmu = PmuResp(fname_resp)

    # Calc pressure
    acq_timestamps = get_acquisition_times(nif_fieldmap)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # Create Coil
    coil_affine = nif_fieldmap.affine
    coil = create_coil(150, 150, nz + 10, create_constraints(np.inf, -np.inf, np.inf), coil_affine)

    # Define the slices to shim with the proper convention
    slices = define_slices(nif_target.shape[2], 5, method='ascending')

    # Find optimal currents
    output = RealTimeSequencer(nif_fieldmap, nif_target, nif_static_mask, nif_riro_mask, slices, pmu,
                               [coil], [coil], method='least_squares').shim()
    currents_static, currents_riro, mean_p, p_rms = output

    # Scale according to rms
    currents_riro_rms = currents_riro * p_rms

    # Print some outputs
    print(f"\nSlices: {slices}"
          f"\nFieldmap affine:\n{nif_fieldmap.affine}\n"
          f"Coil affine:\n{coil_affine}\n"
          f"Static currents:\n{currents_static}\n"
          f"Riro currents * p_rms:\n{currents_riro_rms}\n")

    # Calculate theoretical shimmed map
    # shim
    unshimmed = nif_fieldmap.data
    nii_target = nib.Nifti1Image(nif_fieldmap.data[..., 0], nif_fieldmap.affine, header=nif_fieldmap.header)
    opt = Optimizer([coil], unshimmed[..., 0], nif_fieldmap.affine)
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
        correction_static = np.sum(currents_static[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        # Calculate the riro coil profiles
        riro_profile = np.sum(currents_riro[i_shim] * opt.merged_coils, axis=3, keepdims=False)

        masked_fieldmap[..., i_shim] = resample_mask(nif_static_mask.nii, nii_target, slices[i_shim]).get_fdata()
        for i_t in range(nif_fieldmap.shape[3]):
            # Apply the static and riro correction
            correction_riro = riro_profile * (acq_pressures[i_t] - mean_p)
            shimmed_static[..., i_t, i_shim] = unshimmed[..., i_t] + correction_static
            shimmed_static_riro[..., i_t, i_shim] = shimmed_static[..., i_t, i_shim] + correction_riro
            shimmed_riro[..., i_t, i_shim] = unshimmed[..., i_t] + correction_riro

            # Calculate masked shim
            masked_shim_static[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static[..., i_t, i_shim]
            masked_shim_static_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_static_riro[
                ..., i_t, i_shim]
            masked_shim_riro[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * shimmed_riro[..., i_t, i_shim]
            masked_unshimmed[..., i_t, i_shim] = masked_fieldmap[..., i_shim] * unshimmed[..., i_t]

            # Calculate the sum over the ROI
            sum_shimmed_static = np.sum(np.abs(masked_shim_static[..., i_t, i_shim]))
            sum_shimmed_static_riro = np.sum(np.abs(masked_shim_static_riro[..., i_t, i_shim]))
            sum_shimmed_riro = np.sum(np.abs(masked_shim_riro[..., i_t, i_shim]))
            sum_unshimmed = np.sum(np.abs(masked_unshimmed[..., i_t, i_shim]))

            # Create a 1D list of the sum of the shimmed and unshimmed maps
            shim_trace_static.append(sum_shimmed_static)
            shim_trace_static_riro.append(sum_shimmed_static_riro)
            shim_trace_riro.append(sum_shimmed_riro)
            unshimmed_trace.append(sum_unshimmed)

            assert sum_shimmed_static_riro < sum_unshimmed


def save_nii(nii_fieldmap, coil, opt, nii_mask):
    """Save relevant nifti files"""
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


array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
array = np.repeat(array, 4, 1)
array = np.repeat(array[..., np.newaxis], 1, 2)
array = np.repeat(array[..., np.newaxis], 5, 3)
affine = np.array([[3.342335, -9.593514, 0.173426, 3],
                   [0.083550, 0.202371, 11.829379, 7],
                   [8.295892, 3.863097, -0.189009, 11],
                   [0, 0, 0, 1]])
# array.shape: (2, 8, 1, 5)
nii = nib.Nifti1Image(array, affine)


@pytest.mark.parametrize(
    "nii_4d", [(
            nii,
    )]
)
class TestExtendSlice(object):
    def test_extend_slice_4d_dim1(self, nii_4d):
        nii_out = extend_slice(nii_4d[0], 1, 0)

        assert nii_out.get_fdata().shape == (4, 8, 1, 5)
        assert np.all(np.isclose(nii_out.affine, np.array([[3.342335, -9.593514, 0.173426, -0.342335],
                                                           [0.08355, 0.202371, 11.829379, 6.91645],
                                                           [8.295892, 3.863097, -0.189009, 2.704108],
                                                           [0., 0., 0., 1.]])))

    def test_extend_slice_4d_dim2(self, nii_4d):
        nii_out = extend_slice(nii_4d[0], 1, 1)

        assert nii_out.get_fdata().shape == (2, 10, 1, 5)

    def test_extend_slice_4d_dim3(self, nii_4d):
        nii_out = extend_slice(nii_4d[0], 1, 2)

        assert nii_out.get_fdata().shape == (2, 8, 3, 5)

    def test_extend_slice_3d(self, nii_4d):
        nii_3d = nib.Nifti1Image(nii_4d[0].get_fdata()[..., 0], nii_4d[0].affine)
        nii_out = extend_slice(nii_3d, 1, 2)

        assert nii_out.get_fdata().shape == (2, 8, 3)

    def test_extend_slice_3d_dim1_2slices(self, nii_4d):
        nii_out = extend_slice(nii_4d[0], 2, 2)

        assert nii_out.get_fdata().shape == (2, 8, 5, 5)

    def test_extend_slice_wrong_dim(self, nii_4d):
        nii_2d = nib.Nifti1Image(nii_4d[0].get_fdata()[..., 0, 0], nii_4d[0].affine)
        with pytest.raises(ValueError, match="Unsupported number of dimensions for input array"):
            extend_slice(nii_2d, 1, 2)

    def test_extend_slice_wrong_axis(self, nii_4d):
        with pytest.raises(ValueError, match="Unsupported value for axis"):
            extend_slice(nii_4d[0], 1, 4)


class TestDefineSlices(object):
    def test_define_slices_default_factor(self):
        output = define_slices(5)
        assert np.all(output == [(0,), (1,), (2,), (3,), (4,)])

    def test_define_slices_interleaved_sms2_even(self):
        output = define_slices(12, 2, "interleaved")
        assert np.all(output == [(1, 7),(5, 11), (3, 9), (2, 8), (0, 6), (4, 10)])

    def test_define_slices_interleaved_sms3_even(self):
        output = define_slices(24, 3, "interleaved")
        assert np.all(output == [(1, 9, 17),(3, 11, 19), (7, 15, 23), (5, 13, 21),
                                 (2, 10, 18), (4, 12, 20), (0, 8, 16), (6, 14, 22)])

    def test_define_slices_interleaved_sms2_odd(self):
        output = define_slices(18, 2, "interleaved")
        assert np.all(output == [(1, 10), (3, 12), (5, 14), (7, 16), (0, 9), (2, 11), (4, 13), (6, 15), (8, 17)])

    def test_define_slices_interleaved_sms3_odd(self):
        output = define_slices(21, 3, "interleaved")
        assert np.all(output == [(0, 7, 14), (2, 9, 16), (4, 11, 18), (6, 13, 20),
                                 (1, 8, 15), (3, 10, 17), (5, 12, 19)])

    def test_define_slices_ascending(self):
        output = define_slices(6, 2, "ascending")
        assert np.all(output == [(0, 3), (1, 4), (2, 5)])

    def test_define_slices_descending(self):
        output = define_slices(6, 2, "descending")
        assert np.all(output == [(2, 5), (1, 4), (0, 3)])

    def test_define_slices_volume(self):
        output = define_slices(5, method="volume")
        assert np.all(output == [(0, 1, 2, 3, 4)])

    def test_define_slices_wrong_method(self):
        with pytest.raises(ValueError, match="Not a supported method to define slices"):
            define_slices(6, 2, "abc")

    def test_leftover_slices(self):
        with pytest.raises(ValueError, match="SMS method does not support leftover slices"):
            define_slices(5, 2, "interleaved")

    def test_define_slices_wrong_n_slice(self):
        with pytest.raises(ValueError, match="Number of slices should be greater than 0"):
            define_slices(0, 2, "ascending")


class TestParseSlices(object):
    def setup_method(self):
        fname = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')

        # Open json
        fname_json = fname.split('.nii')[0] + '.json'
        # Read from json file
        with open(fname_json) as json_file:
            json_data = json.load(json_file)

        json_data['SliceTiming'] = [10, 10, 0, 30, 30]
        self.json_data = json_data

    def test_parse_slices(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_json = os.path.join(tmp, 'test.json')
            fname_nifti = os.path.join(tmp, 'test.nii')

            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=4)

            slices = parse_slices(fname_nifti)

            assert slices == [(2,), (0, 1), (3, 4)]

    def test_parse_slices_real_data(self):
        fname = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
        slices = parse_slices(fname)

        assert slices == [(1,), (3,), (5,), (7,), (9,), (11,), (13,), (15,), (17,), (19,),
                          (0,), (2,), (4,), (6,), (8,), (10,), (12,), (14,), (16,), (18,)]

    def test_parse_slices_slice_encode(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_json = os.path.join(tmp, 'test.json')
            fname_nifti = os.path.join(tmp, 'test.nii')

            self.json_data['SliceEncodingDirection'] = 'k-'
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=4)

            slices = parse_slices(fname_nifti)

            assert slices == [(2,), (3, 4), (0, 1)]


class TestMaxintensity():
    """ We are using a 4d fieldmap as input just for testing. """

    def setup_method(self):
        fname_input = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
        self.nii_input = nib.load(fname_input)

        # Set up mask: Cube
        nx, ny, nz = self.nii_input.shape[:3]
        mask = shapes(self.nii_input.get_fdata()[..., 0], 'cube',
                      center_dim1=32,
                      center_dim2=36,
                      len_dim1=10, len_dim2=10, len_dim3=nz)
        self.nii_mask = nib.Nifti1Image(mask.astype(np.uint8), self.nii_input.affine)

    def test_default_max_intensity(self):
        output = shim_max_intensity(self.nii_input, self.nii_mask)
        assert output == 8

    def test_max_intensity_res_mask(self):
        slice = self.nii_input.get_fdata()[:-3, :-3, 0, 0] > 100
        nii_diff_mask = nib.Nifti1Image(np.concatenate((slice[..., np.newaxis], slice[..., np.newaxis]), axis=2),
                                        self.nii_input.affine, header=self.nii_input.header)
        output = shim_max_intensity(self.nii_input, nii_diff_mask)
        assert output == 0

    def test_max_intensity_wrong_input_dim(self):
        with pytest.raises(ValueError, match="Input volume must be 4d"):
            shim_max_intensity(self.nii_mask, self.nii_mask)

    def test_max_intensity_wrong_mask_dim(self):
        with pytest.raises(ValueError, match="Input mask must be 3d"):
            shim_max_intensity(self.nii_input, self.nii_input)
