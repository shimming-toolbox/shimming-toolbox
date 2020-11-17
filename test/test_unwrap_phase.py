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
        unwrapped = unwrap_phase(self.phase, self.mag, self.affine, unwrapper='prelude')
        assert unwrapped.shape == self.phase.shape

    def test_unwrap_phase_prelude_3d(self):
        phase = self.phase[..., 0]
        mag = self.mag[..., 0]
        unwrapped = unwrap_phase(phase, mag, self.affine, unwrapper='prelude')
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_2d(self):
        phase = self.phase[..., 0, 0]
        mag = self.mag[..., 0, 0]
        unwrapped = unwrap_phase(phase, mag, self.affine, unwrapper='prelude')
        assert unwrapped.shape == phase.shape

    def test_unwrap_phase_prelude_threshold(self):
        unwrapped = unwrap_phase(self.phase, self.mag, self.affine, unwrapper='prelude', threshold=0.1)
        assert unwrapped.shape == self.phase.shape

    def test_unwrap_phase_prelude_4d_mask(self):
        unwrapped = unwrap_phase(self.phase, self.mag, self.affine, unwrapper='prelude', mask=np.ones_like(self.phase))
        assert unwrapped.shape == self.phase.shape
