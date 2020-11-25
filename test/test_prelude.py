# coding: utf-8

from pathlib import Path
import shutil
import os
import glob
import nibabel as nib
import numpy as np
import logging
import pytest

from shimmingtoolbox.unwrap import prelude


@pytest.mark.prelude
@pytest.mark.usefixtures("test_prelude_installation")
class TestCore(object):

    def setup(self):
        # Get the directory where this current file is saved
        self.full_path = Path(__file__).resolve().parent
        # "test/" directory
        self.test_path = self.full_path

        # Create temp folder
        # Get the directory where this current file is saved
        self.tmp_path = self.test_path / '__tmp__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()
        self.toolbox_path = self.test_path.parent

        self.get_phases_mags_affines()

    def teardown(self):
        # Remove temporary files
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
            pass

    def get_phases_mags_affines(self):
        """
        Get the phase and mag images (as np.array) and affine matrices from the testing data.
        """
        path_data = glob.glob(os.path.join(self.toolbox_path, 'testing_data*'))[0]

        # Open phase data
        fname_phases = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
        if len(fname_phases) > 2:
            raise IndexError('Phase data parsing is wrongly parsed')

        nii_phase_e1 = nib.load(fname_phases[0])
        nii_phase_e2 = nib.load(fname_phases[1])

        # Scale to phase to radians
        phase_e1 = np.interp(nii_phase_e1.get_fdata(), [0, 4096], [-np.pi, np.pi])
        phase_e2 = np.interp(nii_phase_e2.get_fdata(), [0, 4096], [-np.pi, np.pi])

        # Open mag data
        fname_mags = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*magnitude*.nii.gz'))

        if len(fname_mags) > 2:
            raise IndexError('Mag data parsing is wrongly parsed')

        nii_mag_e1 = nib.load(fname_mags[0])
        nii_mag_e2 = nib.load(fname_mags[1])

        # Make tests fater by having the last dim only be 1
        self.phase_e1 = np.expand_dims(phase_e1[..., 0], -1)
        self.phase_e2 = np.expand_dims(phase_e2[..., 0], -1)
        self.mag_e1 = np.expand_dims(nii_mag_e1.get_fdata()[..., 0], -1)
        self.mag_e2 = np.expand_dims(nii_mag_e2.get_fdata()[..., 0], -1)
        self.affine_phase_e1 = nii_phase_e1.affine
        self.affine_phase_e2 = nii_phase_e2.affine

    def test_default_works(self):
        """
        Runs prelude and check output integrity.
        """
        # default prelude call
        unwrapped_phase_e1 = prelude(self.phase_e1, self.affine_phase_e1)

        assert (unwrapped_phase_e1.shape == self.phase_e1.shape)

    def test_non_default_mask(self):
        """
        Check prelude function with input binary mask.
        """
        # Create mask with all ones
        mask = np.ones(self.phase_e1.shape)

        # Call prelude with mask (is_unwrapping_in_2d is also used because it is significantly faster)
        unwrapped_phase_e1 = prelude(self.phase_e1, self.affine_phase_e1, mag=self.mag_e1, mask=mask,
                                     is_unwrapping_in_2d=True)

        # Make sure the phase is not 0. When there isn't a mask, the phase is 0
        assert(unwrapped_phase_e1[5, 5, 0] != 0)

    def test_wrong_size_mask(self):
        # Create mask with wrong dimensions
        mask = np.ones([4, 4, 4])

        # Call prelude with mask
        try:
            prelude(self.phase_e1, self.affine_phase_e1, mag=self.mag_e1, mask=mask)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour since the mask is the wrong dimensions
            return 0

        # If there isn't an error, then there is a problem
        print('\nWrong dimensions for mask does not throw an error')
        assert False

    def test_wrong_phase_dimensions(self):
        # Call prelude phase with wrong dimensions
        phase_e1 = np.ones([4, 4])

        try:
            prelude(phase_e1, self.affine_phase_e1, mag=self.mag_e1)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print('\nWrong dimensions for phase input')
        assert False

    def test_wrong_mag_dimensions(self):
        # Call prelude phase with wrong dimensions
        mag_e1 = np.ones([4, 4, 4])

        try:
            prelude(self.phase_e1, self.affine_phase_e1, mag=mag_e1)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print('\nWrong dimensions for mag input')
        assert False

    def test_wrong_mag_and_phase_dimensions(self):
        # Call prelude phase with wrong dimensions
        mag_e1 = np.ones([4, 4, 4])
        phase_e1 = np.ones([4])

        try:
            prelude(phase_e1, self.affine_phase_e1, mag=mag_e1)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print('\nWrong dimensions both mag and phase input')
        assert False

    def test_phase_2d(self):
        """
        Call prelude with a 2D phase and mag array
        """
        # Get first slice
        phase_e1_2d = self.phase_e1[:, :, 0]
        mag_e1_2d = self.mag_e1[:, :, 0]
        unwrapped_phase_e1 = prelude(phase_e1_2d, self.affine_phase_e1, mag=mag_e1_2d)

        assert(unwrapped_phase_e1.shape == phase_e1_2d.shape)

    def test_threshold(self):
        """
        Call prelude with a threshold for masking
        """
        unwrapped_phase_e1 = prelude(self.phase_e1, self.affine_phase_e1, mag=self.mag_e1, threshold=200)
        assert(unwrapped_phase_e1.shape == self.phase_e1.shape)

    def test_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the z dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(self.phase_e1[..., 0], -1)
        mag_singleton = np.expand_dims(self.mag_e1[..., 0], -1)

        unwrapped_phase_singleton = prelude(phase_singleton, self.affine_phase_e1, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3

    def test_2nd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(self.phase_e1[:, 0, 0], -1)
        mag_singleton = np.expand_dims(self.mag_e1[:, 0, 0], -1)

        unwrapped_phase_singleton = prelude(phase_singleton, self.affine_phase_e1, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 2

    def test_2nd_and_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd and 3rd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(np.expand_dims(self.phase_e1[:, 0, 0], -1), -1)
        mag_singleton = np.expand_dims(np.expand_dims(self.mag_e1[:, 0, 0], -1), -1)

        unwrapped_phase_singleton = prelude(phase_singleton, self.affine_phase_e1, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3
