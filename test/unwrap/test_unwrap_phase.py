#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import nibabel as nib
import numpy as np
import pytest
import math

from shimmingtoolbox.unwrap.unwrap_phase import unwrap_phase
from shimmingtoolbox import __dir_testing__


@pytest.mark.prelude
class TestUnwrapPhase(object):
    def setup(self):
        fname_phase = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.nii.gz')
        nii_phase = nib.load(fname_phase)
        phase = (nii_phase.get_fdata() * 2 * math.pi / 4095) - math.pi  # [-pi, pi]
        self.nii_phase = nib.Nifti1Image(phase, nii_phase.affine, header=nii_phase.header)
        fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
        nii_mag = nib.load(fname_mag)
        self.mag = nii_mag.get_fdata()

    def test_unwrap_phase_prelude_4d(self):
        """Test prelude with 4d input data."""
        unwrapped = unwrap_phase(self.nii_phase, unwrapper='prelude', mag=self.mag)
        assert unwrapped.shape == self.nii_phase.shape

    def test_unwrap_phase_prelude_3d(self):
        """Test prelude with 3d input data."""
        phase = self.nii_phase.get_fdata()[..., 0]
        nii = nib.Nifti1Image(phase, self.nii_phase.affine, header=self.nii_phase.header)
        mag = self.mag[..., 0]

        unwrapped = unwrap_phase(nii, unwrapper='prelude', mag=mag)
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_2d(self):
        """Test prelude with 2d input data."""
        phase = self.nii_phase.get_fdata()[..., 0, 0]
        nii = nib.Nifti1Image(phase, self.nii_phase.affine, header=self.nii_phase.header)
        mag = self.mag[..., 0, 0]

        unwrapped = unwrap_phase(nii, unwrapper='prelude', mag=mag)
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_threshold(self):
        """Test prelude with threshold parameter."""
        unwrapped = unwrap_phase(self.nii_phase, unwrapper='prelude', mag=self.mag, threshold=0.1)
        assert unwrapped.shape == self.nii_phase.shape

    def test_unwrap_phase_prelude_4d_mask(self):
        """Test prelude with mask parameter."""
        unwrapped = unwrap_phase(self.nii_phase, unwrapper='prelude', mag=self.mag,
                                 mask=np.ones(self.nii_phase.shape))
        assert unwrapped.shape == self.nii_phase.shape

    def test_unwrap_phase_prelude_2d_mask(self):
        """Test prelude with 2d mask parameter."""
        phase = self.nii_phase.get_fdata()[..., 0, 0]
        nii = nib.Nifti1Image(phase, self.nii_phase.affine, header=self.nii_phase.header)
        mag = self.mag[..., 0, 0]

        unwrapped = unwrap_phase(nii, unwrapper='prelude', mag=mag, mask=np.ones_like(phase))
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_wrong_unwrapper(self):
        """Input wrong unwrapper."""

        with pytest.raises(NotImplementedError, match="The unwrap function"):
            unwrap_phase(self.nii_phase, mag=self.mag, unwrapper='Not yet implemented.')

    def test_unwrap_phase_wrong_shape(self):
        """Input wrong shape."""
        phase = np.expand_dims(self.nii_phase.get_fdata(), -1)
        nii = nib.Nifti1Image(phase, self.nii_phase.affine, header=self.nii_phase.header)

        with pytest.raises(ValueError, match="Shape of input phase is not supported."):
            unwrap_phase(nii, mag=self.mag)

    def test_unwrap_phase_skimage(self):
        phase = self.nii_phase.get_fdata()[..., 0, :]
        nii = nib.Nifti1Image(phase, self.nii_phase.affine, header=self.nii_phase.header)
        mag = self.mag[..., 0, :]

        unwrapped_skimage = unwrap_phase(nii, unwrapper='skimage', mag=mag, threshold=100)

        assert unwrapped_skimage.shape == phase.shape
        assert np.allclose(unwrapped_skimage[29, 60:65, 0],
                           [0.11891254, 0.15727143, 0.13272174, 0.16187449, 0.14959965])
