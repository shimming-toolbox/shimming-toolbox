# coding: utf-8

from pathlib import Path
import shutil
import os
import glob
import nibabel as nib
import numpy as np

import pytest

from shimmingtoolbox.unwrap import prelude


class TestCore(object):

    def setup(self):
        # Get the directory where this current file is saved
        self.full_path = Path(__file__).resolve().parent
        print(self.full_path)
        # "test/" directory
        self.test_path = self.full_path

        # Create temp folder
        # Get the directory where this current file is saved
        self.tmp_path = self.test_path / '__tmp__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()
        print(self.tmp_path)
        self.toolbox_path = self.test_path.parent

    def teardown(self):
        # Get the directory where this current file is saved
        print(self.full_path)
        print(self.test_path)
        print(self.tmp_path)

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
            pass

    def get_phases_mags_affines(self):
        path_data = glob.glob(os.path.join(self.toolbox_path, 'testing_data*'))[0]

        # Open phase data
        fname_phases = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
        if len(fname_phases) > 2:
            raise IndexError('Phase data parsing is wrongly parsed')

        nii_phase_e1 = nib.load(fname_phases[0])
        nii_phase_e2 = nib.load(fname_phases[1])

        # Scale to phase to radians
        phase_e1 = nii_phase_e1.get_fdata() / 4096 * 2 * np.pi - np.pi
        phase_e2 = nii_phase_e2.get_fdata() / 4096 * 2 * np.pi - np.pi

        # Open mag data
        fname_mags = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*magnitude*.nii.gz'))

        if len(fname_mags) > 2:
            raise IndexError('Mag data parsing is wrongly parsed')

        nii_mag_e1 = nib.load(fname_mags[0])
        nii_mag_e2 = nib.load(fname_mags[1])

        return phase_e1, phase_e2, nii_mag_e1.get_fdata(), nii_mag_e2.get_fdata(), nii_phase_e1, nii_phase_e2

    def test_prelude_default_works(self):
        # Get the phase, mag and affine matrices
        phase_e1, phase_e2, nii_mag_e1, nii_mag_e2, nii_phase_e1, nii_phase_e2 = self.get_phases_mags_affines()

        # default prelude call
        unwrapped_phase_e1 = prelude(phase_e1, nii_mag_e1, nii_phase_e1.affine)
        unwrapped_phase_e2 = prelude(phase_e2, nii_mag_e2, nii_phase_e1.affine)

        # Compute phase difference
        unwrapped_phase = unwrapped_phase_e2 - unwrapped_phase_e1

        # Compute wrapped phase diff
        wrapped_phase = phase_e2 - phase_e1

        assert (unwrapped_phase.shape == wrapped_phase.shape)

    # TODO: More thorough tests. (eg: Test for all different options, test error handling, etc)

    def test_prelude_non_default_path(self):
        phase_e1, phase_e2, nii_mag_e1, nii_mag_e2, nii_phase_e1, nii_phase_e2 = self.get_phases_mags_affines()

        unwrapped_phase_e1 = prelude(phase_e1, nii_mag_e1, nii_phase_e1.affine,
                                     path_2_unwrapped_phase=os.path.join(self.tmp_path, 'tmp', 'data.nii'),
                                     is_saving_nii=True)

        assert (os.path.exists(os.path.join(self.tmp_path, 'tmp', 'data.nii')))


