#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import os
import pathlib
import pytest
import re
import tempfile

from shimmingtoolbox.unwrap.skimage_unwrap import skimage_unwrap
from . import get_phases_mags

nii_phase_e1, nii_phase_e2, mag_e1, mag_e2 = get_phases_mags()


class TestSkimageUnwrap(object):
    """Test skimage_unwrap."""
    def test_specific_values(self):
        unwrap_phase = skimage_unwrap(nii_phase_e1, mag_e1, threshold=300)
        assert unwrap_phase.shape == nii_phase_e1.shape
        assert np.allclose(unwrap_phase[64, 40:45, 0], [0.19021362, 0.10277671, 0.02914563, -0.06442719, -0.15800002])

    def test_default(self):
        unwrap_phase = skimage_unwrap(nii_phase_e1)
        assert unwrap_phase.shape == nii_phase_e1.shape

    def test_output_mask(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname = os.path.join(tmp, 'mask.nii.gz')
            skimage_unwrap(nii_phase_e1, fname_save_mask=fname)
            assert os.path.isfile(fname)

    def test_wrong_phase_dim(self):
        nii = nib.Nifti1Image(np.ones([3]), nii_phase_e1.affine, header=nii_phase_e1.header)
        with pytest.raises(ValueError, match="Wrapped_phase must be 2d or 3d"):
            skimage_unwrap(nii)

    def test_wrong_mask_dim(self):
        mask = np.ones([4])
        with pytest.raises(ValueError, match="Mask must be the same shape as wrapped_phase"):
            skimage_unwrap(nii_phase_e1, mask=mask)

    def test_wrong_mag_dim(self):
        mag = np.ones([4])
        with pytest.raises(ValueError,
                           match=re.escape("The magnitude image (mag) must be the same shape as wrapped_phase")):
            skimage_unwrap(nii_phase_e1, mag=mag)

    def test_mag_no_threshold(self):
        with pytest.raises(ValueError, match="Threshold must be specified if a mag is provided"):
            skimage_unwrap(nii_phase_e1, mag_e1)
