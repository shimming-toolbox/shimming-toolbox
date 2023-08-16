#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import pytest

from shimmingtoolbox.unwrap import prelude
from . import get_phases_mags

nii_phase_e1, nii_phase_e2, mag_e1, mag_e2 = get_phases_mags()


@pytest.mark.prelude
# @pytest.mark.usefixtures("test_prelude_installation")
class TestPrelude(object):

    def test_default_works(self):
        """
        Runs prelude and check output integrity.
        """
        # default prelude call
        unwrapped_phase_e1 = prelude(nii_phase_e1)
        assert unwrapped_phase_e1.shape == nii_phase_e1.shape

    def test_non_default_mask(self):
        """
        Check prelude function with input binary mask.
        """
        # Create mask with all ones
        mask = np.ones(nii_phase_e1.shape)

        # Call prelude with mask (is_unwrapping_in_2d is also used because it is significantly faster)
        unwrapped_phase_e1 = prelude(nii_phase_e1, mag=mag_e1, mask=mask, is_unwrapping_in_2d=True)

        # Make sure the phase is not 0. When there isn't a mask, the phase is 0
        assert unwrapped_phase_e1[5, 5, 0] != 0

    def test_wrong_size_mask(self):
        # Create mask with wrong dimensions
        mask = np.ones([4, 4, 4])

        with pytest.raises(ValueError, match="Mask must be the same shape as wrapped_phase"):
            prelude(nii_phase_e1, mag=mag_e1, mask=mask)

    def test_wrong_phase_dimensions(self):
        # Call prelude phase with wrong dimensions
        phase_e1 = np.ones([4])
        nii = nib.Nifti1Image(phase_e1, nii_phase_e1.affine, header=nii_phase_e1.header)

        with pytest.raises(ValueError, match="Wrapped_phase must be 2d or 3d"):
            prelude(nii, mag=mag_e1)

    def test_wrong_mag_dimensions(self):
        # Call prelude phase with wrong dimensions
        mag_e1 = np.ones([4, 4, 4])

        with pytest.raises(ValueError, match=r"The magnitude image \(mag\) must be the same shape as wrapped_phase"):
            prelude(nii_phase_e1, mag=mag_e1)

    def test_phase_2d(self):
        """
        Call prelude with a 2D phase and mag array
        """
        # Get first slice
        phase_e1_2d = nii_phase_e1.get_fdata()[:, :, 0]
        mag_e1_2d = mag_e1[:, :, 0]
        nii = nib.Nifti1Image(phase_e1_2d, nii_phase_e1.affine, header=nii_phase_e1.header)

        unwrapped_phase_e1 = prelude(nii, mag=mag_e1_2d)

        assert unwrapped_phase_e1.shape == phase_e1_2d.shape

    def test_threshold(self):
        """
        Call prelude with a threshold for masking
        """
        unwrapped_phase_e1 = prelude(nii_phase_e1, mag=mag_e1, threshold=200)
        assert unwrapped_phase_e1.shape == nii_phase_e1.shape

    def test_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the z dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(nii_phase_e1.get_fdata()[..., 0], -1)
        mag_singleton = np.expand_dims(mag_e1[..., 0], -1)
        nii = nib.Nifti1Image(phase_singleton, nii_phase_e1.affine, header=nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3

    def test_2nd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(nii_phase_e1.get_fdata()[:, 0, 0], -1)
        mag_singleton = np.expand_dims(mag_e1[:, 0, 0], -1)
        nii = nib.Nifti1Image(phase_singleton, nii_phase_e1.affine, header=nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 2

    def test_2nd_and_3rd_dim_singleton(self):
        """
        Call prelude on data with a singleton on the 2nd and 3rd dimension
        """

        # Prepare singleton
        phase_singleton = np.expand_dims(np.expand_dims(nii_phase_e1.get_fdata()[:, 0, 0], -1), -1)
        mag_singleton = np.expand_dims(np.expand_dims(mag_e1[:, 0, 0], -1), -1)
        nii = nib.Nifti1Image(phase_singleton, nii_phase_e1.affine, header=nii_phase_e1.header)

        unwrapped_phase_singleton = prelude(nii, mag=mag_singleton)

        assert unwrapped_phase_singleton.ndim == 3
