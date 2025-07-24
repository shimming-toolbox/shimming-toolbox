import os
import tempfile

import nibabel as nib
import numpy as np

from shimmingtoolbox.masking.softmasks import create_softmask, convert_to_pixels, save_softmask

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

class TestUtilityFunctions:
    def setup_method(self):
        # Create a dummy binary mask to extract header and test save
        self.binary_mask = np.zeros((5, 5, 5), dtype=np.float32)
        self.binary_mask[2, 2, 2] = 1.0
        self.nii = nib.Nifti1Image(self.binary_mask, affine=np.eye(4))
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path_binmask = os.path.join(self.tmpdir.name, 'binmask.nii.gz')
        nib.save(self.nii, self.path_binmask)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_convert_to_pixels_mm(self):
        header = self.nii.header
        header.set_zooms((2.0, 2.0, 2.0))  # voxel size of 2 mm
        pixels = convert_to_pixels(6, 'mm', header)
        assert pixels == 3  # 6 mm / 2 mm = 3 pixels

    def test_convert_to_pixels_px(self):
        header = self.nii.header
        pixels = convert_to_pixels(5, 'px', header)
        assert pixels == 5

    def test_convert_to_pixels_invalid_unit(self):
        header = self.nii.header
        try:
            convert_to_pixels(6, 'cm', header)
            assert False, "Expected ValueError for invalid unit"
        except ValueError as e:
            assert "Lenght must be" in str(e)

    def test_save_softmask(self):
        soft_mask = np.ones_like(self.binary_mask, dtype=np.float32)
        path_output = os.path.join(self.tmpdir.name, 'softmask.nii.gz')
        nii_out = save_softmask(soft_mask, path_output, self.path_binmask)
        assert os.path.exists(path_output)
        loaded = nib.load(path_output).get_fdata()
        assert np.allclose(loaded, soft_mask)
