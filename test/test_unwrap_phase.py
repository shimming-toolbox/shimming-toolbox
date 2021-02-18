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
        fname_phase = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                   'sub-example_phasediff.nii.gz')
        nii_phase = nib.load(fname_phase)
        self.phase = (nii_phase.get_fdata() * 2 * math.pi / 4095) - math.pi  # [-pi, pi]
        self.affine = nii_phase.affine
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        nii_mag = nib.load(fname_mag)
        self.mag = nii_mag.get_fdata()

    def test_unwrap_phase_prelude_4d(self):
        """Test prelude with 4d input data."""
        unwrapped = unwrap_phase(self.phase, self.affine, unwrapper='prelude', mag=self.mag)
        assert unwrapped.shape == self.phase.shape

    def test_unwrap_phase_prelude_3d(self):
        """Test prelude with 3d input data."""
        phase = self.phase[..., 0]
        mag = self.mag[..., 0]
        unwrapped = unwrap_phase(phase, self.affine, unwrapper='prelude', mag=mag)
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_2d(self):
        """Test prelude with 2d input data."""
        phase = self.phase[..., 0, 0]
        mag = self.mag[..., 0, 0]
        unwrapped = unwrap_phase(phase, self.affine, unwrapper='prelude', mag=mag)
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_threshold(self):
        """Test prelude with threshold parameter."""
        unwrapped = unwrap_phase(self.phase, self.affine, unwrapper='prelude', mag=self.mag, threshold=0.1)
        assert unwrapped.shape == self.phase.shape

    def test_unwrap_phase_prelude_4d_mask(self):
        """Test prelude with mask parameter."""
        unwrapped = unwrap_phase(self.phase, self.affine, unwrapper='prelude', mag=self.mag,
                                 mask=np.ones_like(self.phase))
        assert unwrapped.shape == self.phase.shape

    def test_unwrap_phase_prelude_2d_mask(self):
        """Test prelude with 2d mask parameter."""
        phase = self.phase[..., 0, 0]
        mag = self.mag[..., 0, 0]
        unwrapped = unwrap_phase(phase, self.affine, unwrapper='prelude', mag=mag, mask=np.ones_like(phase))
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_wrong_unwrapper(self):
        """Input wrong unwrapper."""

        # This should return an error
        try:
            unwrap_phase(self.phase, self.affine, mag=self.mag, unwrapper='Not yet implemented.')
        except NotImplementedError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print('\nNot supported unwrapper but does not throw an error.')
        assert False

    def test_unwrap_phase_wrong_shape(self):
        """Input wrong shape."""

        # This should return an error
        try:
            unwrap_phase(np.expand_dims(self.phase, -1), self.affine, mag=self.mag)
        except ValueError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print('\nWrong dimensions but does not throw an error.')
        assert False
