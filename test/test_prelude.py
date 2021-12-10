# coding: utf-8

from pathlib import Path
import shutil
import os
import glob
import nibabel as nib
import numpy as np
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
        fname_phases = glob.glob(os.path.join(path_data, 'ds_b0', 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
        if len(fname_phases) > 2:
            raise IndexError('Phase data parsing is wrongly parsed')

        nii_phase_e1 = nib.load(fname_phases[0])
        nii_phase_e2 = nib.load(fname_phases[1])

        # Scale to phase to radians
        phase_e1 = np.interp(nii_phase_e1.get_fdata(), [0, 4096], [-np.pi, np.pi])
        phase_e2 = np.interp(nii_phase_e2.get_fdata(), [0, 4096], [-np.pi, np.pi])

        # Open mag data
        fname_mags = glob.glob(os.path.join(path_data, 'ds_b0', 'sub-fieldmap', 'fmap', '*magnitude*.nii.gz'))

        if len(fname_mags) > 2:
            raise IndexError('Mag data parsing is wrongly parsed')

        nii_mag_e1 = nib.load(fname_mags[0])
        nii_mag_e2 = nib.load(fname_mags[1])

        # Make tests faster by having the last dim only be 1
        self.nii_phase_e1 = nib.Nifti1Image(np.expand_dims(phase_e1[..., 0], -1), nii_phase_e1.affine,
                                            header=nii_phase_e1.header)
        self.nii_phase_e2 = nib.Nifti1Image(np.expand_dims(phase_e2[..., 0], -1), nii_phase_e1.affine,
                                            header=nii_phase_e1.header)
        self.mag_e1 = np.expand_dims(nii_mag_e1.get_fdata()[..., 0], -1)
        self.mag_e2 = np.expand_dims(nii_mag_e2.get_fdata()[..., 0], -1)

    def test_default_works(self):
        """
        Runs prelude and check output integrity.
        """
        # default prelude call
        unwrapped_phase_e1 = prelude(self.nii_phase_e1)

        assert (unwrapped_phase_e1.shape == self.nii_phase_e1.shape)

    def test_non_default_mask(self):
        """
        Check prelude function with input binary mask.
        """
        # Create mask with all ones
        mask = np.ones(self.nii_phase_e1.shape)

        # Call prelude with mask (is_unwrapping_in_2d is also used because it is significantly faster)
        unwrapped_phase_e1 = prelude(self.nii_phase_e1, mag=self.mag_e1, mask=mask,
                                     is_unwrapping_in_2d=True)

        # Make sure the phase is not 0. When there isn't a mask, the phase is 0
        assert(unwrapped_phase_e1[5, 5, 0] != 0)

    def test_wrong_size_mask(self):
        # Create mask with wrong dimensions
        mask = np.ones([4, 4, 4])

        with pytest.raises(ValueError, match="Mask must be the same shape as wrapped_phase"):
            prelude(self.nii_phase_e1, mag=self.mag_e1, mask=mask)

    def test_wrong_phase_dimensions(self):
        # Call prelude phase with wrong dimensions
        phase_e1 = np.ones([4])
        nii = nib.Nifti1Image(phase_e1, self.nii_phase_e1.affine, header=self.nii_phase_e1.header)

        with pytest.raises(ValueError, match="Wrapped_phase must be 2d or 3d"):
            prelude(nii, mag=self.mag_e1)

    def test_wrong_mag_dimensions(self):
        # Call prelude phase with wrong dimensions
        mag_e1 = np.ones([4, 4, 4])

        with pytest.raises(ValueError, match=r"The magnitude image \(mag\) must be the same shape as wrapped_phase"):
            prelude(self.nii_phase_e1, mag=mag_e1)

    def test_phase_2d(self):
        """
        Call prelude with a 2D phase and mag array
        """
        # Get first slice
        phase_e1_2d = self.nii_phase_e1.get_fdata()[:, :, 0]
        mag_e1_2d = self.mag_e1[:, :, 0]
        nii = nib.Nifti1Image(phase_e1_2d, self.nii_phase_e1.affine, header=self.nii_phase_e1.header)

        unwrapped_phase_e1 = prelude(nii, mag=mag_e1_2d)

        assert(unwrapped_phase_e1.shape == phase_e1_2d.shape)

    def test_threshold(self):
        """
        Call prelude with a threshold for masking
        """
        unwrapped_phase_e1 = prelude(self.nii_phase_e1, mag=self.mag_e1, threshold=200)
        assert(unwrapped_phase_e1.shape == self.nii_phase_e1.shape)

    def test_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the z dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(self.nii_phase_e1.get_fdata()[..., 0], -1)
        mag_singleton = np.expand_dims(self.mag_e1[..., 0], -1)
        nii = nib.Nifti1Image(phase_singleton, self.nii_phase_e1.affine, header=self.nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3

    def test_2nd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(self.nii_phase_e1.get_fdata()[:, 0, 0], -1)
        mag_singleton = np.expand_dims(self.mag_e1[:, 0, 0], -1)
        nii = nib.Nifti1Image(phase_singleton, self.nii_phase_e1.affine, header=self.nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 2

    def test_2nd_and_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd and 3rd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(np.expand_dims(self.nii_phase_e1.get_fdata()[:, 0, 0], -1), -1)
        mag_singleton = np.expand_dims(np.expand_dims(self.mag_e1[:, 0, 0], -1), -1)
        nii = nib.Nifti1Image(phase_singleton, self.nii_phase_e1.affine, header=self.nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3
