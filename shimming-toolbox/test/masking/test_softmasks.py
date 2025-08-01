import os
import tempfile

import nibabel as nib
import numpy as np

from shimmingtoolbox.masking.softmasks import create_softmask, save_softmask

class TestSoftmaskCreation:
    def setup_method(self):
        # Minimal binary mask
        self.binmask = np.zeros((10, 10, 10))
        self.binmask[4:6, 4:6, 4:6] = 1
        self.nii_binmask = nib.Nifti1Image(self.binmask.astype(np.float32), affine=np.eye(4))
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path_binmask = os.path.join(self.tmpdir.name, 'binmask.nii.gz')
        nib.save(self.nii_binmask, self.path_binmask)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def check_softmask(self, softmask):
        # Dimensions equal
        assert softmask.shape == self.binmask.shape, "The soft mask has incorrect dimensions"
        # Values in [0, 1]
        assert np.min(softmask) >= 0.0 and np.max(softmask) <= 1.0, "The soft mask values are out of range"
        # Binary region unchanged
        assert np.array_equal((softmask == 1.0), self.binmask.astype(bool)), "Mismatch in binary regions between binmask and softmask"

    def test_create_two_levels_softmask(self):
        """Test for the creation of a 2 levels soft mask"""
        softmask = create_softmask(self.path_binmask, type='2levels', soft_width=6, width_unit='mm', soft_value=0.5)
        self.check_softmask(softmask)

    def test_create_linear_softmask(self):
        """Test for the creation of a linear soft mask"""
        softmask = create_softmask(self.path_binmask, type='linear', soft_width=6, width_unit='mm')
        self.check_softmask(softmask)

    def test_create_gaussian_softmask(self):
        """Test for the creation of a gaussian soft mask"""
        softmask = create_softmask(self.path_binmask, type='gaussian', soft_width=6, width_unit='mm')
        self.check_softmask(softmask)

    def test_create_sum_softmask(self):
        """Test for the creation of a summed soft mask"""
        gaussmask = create_softmask(self.path_binmask, type='gaussian', soft_width=6, width_unit='mm')
        # Save gaussmask as NIfTI
        nii_gaussmask = nib.Nifti1Image(gaussmask.astype(np.float32), affine=np.eye(4))
        self.path_gaussmask = os.path.join(self.tmpdir.name, 'gaussmask.nii.gz')
        nib.save(nii_gaussmask, self.path_gaussmask)
        softmask = create_softmask(self.path_binmask, fname_softmask=self.path_gaussmask, type='sum')
        self.check_softmask(softmask)

    def test_create_softmask_invalid_unit(self):
        """Test for invalid width unit"""
        try:
            create_softmask(self.path_binmask, type='2levels', soft_width=6, width_unit='invalid')
        except ValueError as e:
            assert str(e) == "Lenght must be 'mm' or 'px'"
        else:
            assert False

    def test_save_softmask(self):
        soft_mask = np.ones_like(self.binmask, dtype=np.float32)
        path_output = os.path.join(self.tmpdir.name, 'softmask.nii.gz')
        nii_out = save_softmask(soft_mask, path_output, self.path_binmask)
        assert os.path.exists(path_output)
        loaded = nib.load(path_output).get_fdata()
        assert np.allclose(loaded, soft_mask)
