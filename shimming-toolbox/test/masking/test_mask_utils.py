#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import nibabel as nib
import numpy as np
import pytest

from shimmingtoolbox.masking.mask_utils import modify_binary_mask, resample_mask
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
    nii_target_fm= nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine, header=nii_fieldmap.header)

    # target image
    fname_target = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
    nii_target = nib.load(fname_target)

    # Set up mask
    # static
    nx, ny, nz = nii_target.shape
    static_mask = shapes(nii_target.get_fdata(), 'cube',
                         center_dim1=int(nx / 2),
                         center_dim2=int(ny / 2),
                         len_dim1=5, len_dim2=5, len_dim3=nz)

    nii_mask_static = nib.Nifti1Image(static_mask.astype(int), nii_target.affine, header=nii_target.header)

    nii_mask_res = resample_mask(nii_mask_static, nii_target_fm, (0,), dilation_kernel='line')

    expected = np.full_like(nii_target_fm.get_fdata(), fill_value=False)
    expected[24:28, 27:29, 0] = 1

    assert np.all(nii_mask_res.get_fdata() == expected)
