#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import nibabel as nib
import numpy as np
import os
import pathlib
import pytest
import shutil
import tempfile

from shimmingtoolbox.cli.maths import maths_cli
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.load_nifti import read_nii

fname_input = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')


def test_mean():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'mean.nii.gz')
        result = runner.invoke(maths_cli, ['mean',
                                           '--input', fname_input,
                                           '--axis', '1',
                                           '--output', fname_output], catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert nib.load(fname_output).shape == (128, 20)


def test_mean_axis_out_of_bound():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'mean.nii.gz')
        with pytest.raises(IndexError, match="axis 3 is out of bounds for array of dimension 3"):
            runner.invoke(maths_cli, ['mean',
                                      '--input', fname_input,
                                      '--axis', '3',
                                      '--output', fname_output], catch_exceptions=False)


class TestMagPhase():
    fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    fname_mag_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.json')
    fname_phase_not_scaled = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap',
                                          'sub-fieldmap_phase1.nii.gz')
    fname_phase_not_scaled_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap',
                                               'sub-fieldmap_phase1.json')
    fname_phase = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1_scaled.nii.gz')
    fname_real = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_real1.nii.gz')
    fname_real_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_real1.json')
    fname_im = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_imaginary1.nii.gz')
    fname_complex = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_complex.nii.gz')
    fname_complex_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_complex.json')
    fname_output = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'output.nii.gz')
    fname_output_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'output.json')

    def setup_method(self):
        # Scale from -pi to pi
        nii_phase, _, _ = read_nii(self.fname_phase_not_scaled)
        nib.save(nii_phase, self.fname_phase)

        # Load mag
        nii_mag = nib.load(self.fname_mag)
        # Calculate real and imaginary data
        real_data = nii_mag.get_fdata() * np.cos(nii_phase.get_fdata())
        nii_real = nib.Nifti1Image(real_data,
                                   nii_mag.affine,
                                   header=nii_mag.header)
        nib.save(nii_real, self.fname_real)
        im_data = nii_mag.get_fdata() * np.sin(nii_phase.get_fdata())
        nii_im = nib.Nifti1Image(im_data,
                                 nii_mag.affine,
                                 header=nii_mag.header)
        nib.save(nii_im, self.fname_im)
        complex_data = np.empty(nii_mag.shape, dtype=np.complex64)
        complex_data.real = real_data
        complex_data.imag = im_data
        nii_complex = nib.Nifti1Image(complex_data,
                                      nii_mag.affine, dtype=np.complex64)
        nib.save(nii_complex, self.fname_complex)

    def teardown_method(self):
        # Delete temporary test files between each tests
        if os.path.exists(self.fname_real):
            os.remove(self.fname_real)
        if os.path.exists(self.fname_im):
            os.remove(self.fname_im)
        if os.path.exists(self.fname_output):
            os.remove(self.fname_output)
        if os.path.exists(self.fname_output_json):
            os.remove(self.fname_output_json)
        if os.path.exists(self.fname_real_json):
            os.remove(self.fname_real_json)
        if os.path.exists(self.fname_complex):
            os.remove(self.fname_complex)
        if os.path.exists(self.fname_complex_json):
            os.remove(self.fname_complex_json)
        # This is the scaled version we created in the setup method
        if os.path.exists(self.fname_phase):
            os.remove(self.fname_phase)

    def test_phase(self):
        shutil.copy(self.fname_mag_json, self.fname_real_json)

        runner = CliRunner()
        result = runner.invoke(maths_cli, ['phase',
                                           '--real', self.fname_real,
                                           '--imaginary', self.fname_im,
                                           '--output', self.fname_output], catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(self.fname_output)
        assert np.all(np.isclose(nib.load(self.fname_output).get_fdata()[56:74, 20:45],
                                 nib.load(self.fname_phase).get_fdata()[56:74, 20:45],
                                 rtol=1e-04, atol=1e-04))

    def test_mag(self):
        shutil.copy(self.fname_phase_not_scaled_json, self.fname_real_json)
        runner = CliRunner()
        result = runner.invoke(maths_cli, ['mag',
                                           '--real', self.fname_real,
                                           '--imaginary', self.fname_im,
                                           '--output', self.fname_output], catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(self.fname_output)
        assert np.all(np.isclose(nib.load(self.fname_output).get_fdata()[56:74, 20:45],
                                 nib.load(self.fname_mag).get_fdata()[56:74, 20:45],
                                 rtol=1e-04, atol=1e-04))

    def test_mag_from_complex(self):
        shutil.copy(self.fname_mag_json, self.fname_complex_json)
        runner = CliRunner()
        result = runner.invoke(maths_cli, ['mag',
                                           '--complex', self.fname_complex,
                                           '--output', self.fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(self.fname_output)
        assert np.all(np.isclose(nib.load(self.fname_output).get_fdata()[56:74, 20:45],
                                 nib.load(self.fname_mag).get_fdata()[56:74, 20:45],
                                 rtol=1e-04, atol=1e-04))

    def test_phase_from_complex(self):
        shutil.copy(self.fname_mag_json, self.fname_complex_json)
        runner = CliRunner()
        result = runner.invoke(maths_cli, ['phase',
                                           '--complex', self.fname_complex,
                                           '--output', self.fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(self.fname_output)
        assert np.all(np.isclose(nib.load(self.fname_output).get_fdata()[56:74, 20:45],
                                 nib.load(self.fname_phase).get_fdata()[56:74, 20:45],
                                 rtol=1e-04, atol=1e-04))
