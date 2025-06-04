#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from shimmingtoolbox.masking.mask_utils import modify_binary_mask, resample_mask, create_softmask
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__

dummy_image = np.zeros([10, 10, 10])
dummy_image[2, 2, 3] = 1
dummy_image[7:10, 2, 5] = 1
dummy_image[2, 6:8, 6:8] = 1

@pytest.mark.parametrize(
    "input_mask,", [(
        dummy_image,
    )]
)
class TestDilateBinaryMask(object):
    def test_dilate_binary_mask_cross(self, input_mask):
        """Default is the cross"""
        dilated = modify_binary_mask(input_mask[0], shape='cross', operation='dilate')

        # Expected slice 7
        expected_slice = np.zeros([10, 10])

        expected_slice[2, 1:4] = 1
        expected_slice[1:4, 2] = 1

        assert np.all(expected_slice == dilated[..., 3])

    def test_dilate_binary_mask_sphere(self, input_mask):

        dilated = modify_binary_mask(input_mask[0], shape='sphere', size=5, operation='dilate')

        # Expected slice 8
        expected_slice = np.zeros([10, 10])
        expected_slice[2, 5:9] = 1
        expected_slice[1:4, 6:8] = 1

        assert np.all(expected_slice == dilated[..., 8])

    def test_dilate_binary_mask_cube(self, input_mask):

        dilated = modify_binary_mask(input_mask[0], shape='cube', operation='dilate')

        # Expected slice 7
        expected_slice = np.zeros([10, 10])
        expected_slice[1:4, 5:9] = 1

        assert np.all(expected_slice == dilated[..., 7])

    def test_dilate_binary_mask_line(self, input_mask):

        dilated = modify_binary_mask(input_mask[0], shape='line', operation='dilate')

        # Expected slice in x,z plane 2
        expected_slice = np.zeros([10, 10])
        expected_slice[7:10, 4:7] = 1
        expected_slice[2, 2:5] = 1
        expected_slice[1:4, 3] = 1

        assert np.all(expected_slice == dilated[:, 2, :])

    def test_erode_binary_mask_cube(self, input_mask):

        dilated = modify_binary_mask(input_mask[0], shape='cube', operation='erode')

        # Expected slice in x,z plane 2
        expected_slice = np.zeros([10, 10])

        assert np.all(expected_slice == dilated[:, 2, :])

    def test_modify_binary_mask_none(self, input_mask):

        dilated = modify_binary_mask(input_mask[0], shape='None', operation='dilate')

        assert np.all(input_mask[0] == dilated)

    def test_modify_binary_mask_wrong_size(self, input_mask):

        with pytest.raises(ValueError, match="Size must be odd and greater or equal to 3"):
            modify_binary_mask(input_mask[0], size=4, operation='dilate')

    def test_modify_binary_mask_wrong_shape(self, input_mask):

        with pytest.raises(ValueError, match="Use of non supported algorithm for dilating the mask"):
            modify_binary_mask(input_mask[0], 'abc', operation='dilate')


def test_resample_mask():
    """Test for function that resamples a mask"""
    # Fieldmap
    fname_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
    nii_fieldmap = nib.load(fname_fieldmap)
    nii_target = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine, header=nii_fieldmap.header)

    # anat image
    fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)

    # Set up mask
    # static
    nx, ny, nz = nii_anat.shape
    static_mask = shapes(nii_anat.get_fdata(), 'cube',
                         center_dim1=int(nx / 2),
                         center_dim2=int(ny / 2),
                         len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nii_anat.affine, header=nii_anat.header)

    nii_mask_res = resample_mask(nii_mask_static, nii_target, (0,), dilation_kernel='line')

    expected = np.full_like(nii_target.get_fdata(), fill_value=False)
    expected[24:28, 27:29, 0] = 1

    assert np.all(nii_mask_res.get_fdata() == expected)

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
        softmask = create_softmask(self.path_binmask, type='2levels', soft_width=6, soft_units='mm', soft_value=0.5)
        self.check_softmask(softmask)

    def test_create_linear_softmask(self):
        """Test for the creation of a linear soft mask"""
        softmask = create_softmask(self.path_binmask, type='linear', soft_width=6, soft_units='mm')
        self.check_softmask(softmask)

    def test_create_gaussian_softmask(self):
        """Test for the creation of a gaussian soft mask"""
        softmask = create_softmask(self.path_binmask, type='gaussian', soft_width=6, soft_units='mm')
        self.check_softmask(softmask)

    def test_create_sum_softmask(self):
        """Test for the creation of a summed soft mask"""
        gaussmask = create_softmask(self.path_binmask, type='gaussian', soft_width=6, soft_units='mm')
        # Save gaussmask as NIfTI
        nii_gaussmask = nib.Nifti1Image(gaussmask.astype(np.float32), affine=np.eye(4))
        self.path_gaussmask = os.path.join(self.tmpdir.name, 'gaussmask.nii.gz')
        nib.save(nii_gaussmask, self.path_gaussmask)
        softmask = create_softmask(self.path_binmask, fname_softmask=self.path_gaussmask, type='sum')
        self.check_softmask(softmask)
