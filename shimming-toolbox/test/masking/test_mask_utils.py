#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import nibabel as nib
import numpy as np
import pytest

from shimmingtoolbox.masking.mask_utils import modify_binary_mask, resample_mask, basic_softmask, linear_softmask, gaussian_filter_softmask, gaussian_sct_softmask, save_softmask
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


@pytest.mark.parametrize("path_sct_binmask, path_sct_softmask", [
    (os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'binmask_sub-01_t2.nii.gz'),
     os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'softmask_basic_sub-01_t2.nii.gz'))])
def test_basic_softmask(path_sct_binmask, path_sct_softmask):
    """ Test for the creation of a basic soft mask """

    # Verify that the binary mask exists
    assert os.path.exists(path_sct_binmask), "The binary mask does not exist"
    # Load the binary mask
    binmask_nifti = nib.load(path_sct_binmask)
    binmask = binmask_nifti.get_fdata()

    # Verifiy that the output folder exists
    assert os.path.exists(os.path.dirname(path_sct_softmask)), "The output folder does not exist"
    # Create and load the basic soft mask
    b_softmask = basic_softmask(path_sct_binmask, 7, 0.5)
    softmask_nifti = save_softmask(b_softmask, path_sct_softmask, path_sct_binmask)
    softmask = softmask_nifti.get_fdata()

    # Verify that the soft mask has been created
    assert os.path.exists(path_sct_softmask), "The soft mask has not been created"
    # Verify that the soft mask has the correct dimensions and values
    assert softmask.shape == binmask.shape, "The soft mask has incorrect dimensions"
    assert np.array_equal(binmask_nifti.affine, softmask_nifti.affine), "The affine matrices do not match."
    assert 0.0 <= np.min(softmask) and np.max(softmask) <= 1.0, "The soft mask values are out of range"
    assert np.array_equal((softmask == 1.0), binmask.astype(bool)), "Mismatch in binary regions between binmask and softmask"


@pytest.mark.parametrize("path_sct_binmask, path_sct_softmask", [
    (os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'binmask_sub-01_t2.nii.gz'),
     os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'softmask_linear_sub-01_t2.nii.gz'))])
def test_linear_softmask(path_sct_binmask, path_sct_softmask):
    """ Test for the creation of a linear soft mask """

    # Verify that the binary mask exists
    assert os.path.exists(path_sct_binmask), "The binary mask does not exist"
    # Load the binary mask
    binmask_nifti = nib.load(path_sct_binmask)
    binmask = binmask_nifti.get_fdata()

    # Verify that the output folder exists
    assert os.path.exists(os.path.dirname(path_sct_softmask)), "The output folder does not exist"
    # Create and load the linear soft mask
    l_softmask = linear_softmask(path_sct_binmask, 7)
    softmask_nifti = save_softmask(l_softmask, path_sct_softmask, path_sct_binmask)
    softmask = softmask_nifti.get_fdata()

    # Verify that the soft mask has been created
    assert os.path.exists(path_sct_softmask), "The soft mask has not been created"
    # Verify that the soft mask has the correct dimensions and values
    assert softmask.shape == binmask.shape, "The soft mask has incorrect dimensions"
    assert np.array_equal(binmask_nifti.affine, softmask_nifti.affine), "The affine matrices do not match."
    assert 0.0 <= np.min(softmask) and np.max(softmask) <= 1.0, "The soft mask values are out of range"
    assert np.array_equal((softmask == 1.0), binmask.astype(bool)), "Mismatch in binary regions between binmask and softmask"


@pytest.mark.parametrize("path_sct_binmask, path_sct_softmask", [
   (os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'binmask_sub-01_t2.nii.gz'),
    os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'softmask_gaussfilt_sub-01_t2.nii.gz'))])
def test_gaussian_filter_softmask(path_sct_binmask, path_sct_softmask):
    """ Test for the creation of a gaussian soft mask """

    # Verify that the binary mask exists
    assert os.path.exists(path_sct_binmask), "The binary mask does not exist"
    # Load the binary mask
    binmask_nifti = nib.load(path_sct_binmask)
    binmask = binmask_nifti.get_fdata()

    # Verify that the output folder exists
    assert os.path.exists(os.path.dirname(path_sct_softmask)), "The output folder does not exist"
    # Create and load the gaussian soft mask
    g_softmask = gaussian_filter_softmask(path_sct_binmask, 7)
    softmask_nifti = save_softmask(g_softmask, path_sct_softmask, path_sct_binmask)
    softmask = softmask_nifti.get_fdata()

    # Verify that the soft mask has been created
    assert os.path.exists(path_sct_softmask), "The soft mask has not been created"
    # Verify that the soft mask has the correct dimensions and values
    assert softmask.shape == binmask.shape, "The soft mask has incorrect dimensions"
    assert np.array_equal(binmask_nifti.affine, softmask_nifti.affine), "The affine matrices do not match."
    assert 0.0 <= np.min(softmask) and np.max(softmask) <= 1.0, "The soft mask values are out of range"
    assert np.array_equal((softmask == 1.0), binmask.astype(bool)), "Mismatch in binary regions between binmask and softmask"


@pytest.mark.parametrize("path_sct_binmask, path_sct_gaussmask, path_sct_softmask", [
   (os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'binmask_sub-01_t2.nii.gz'),
    os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'gaussmask_sub-01_t2.nii.gz'),
    os.path.join(__dir_testing__, 'ds_spine', 'derivatives', 'ds_spine_masks', 'softmask_gaussct_sub-01_t2.nii.gz'))])
def test_gaussian_sct_softmask(path_sct_binmask, path_sct_gaussmask, path_sct_softmask):
    """ Test for the creation of a gaussian soft mask """

    # Verify that the binary mask exists
    assert os.path.exists(path_sct_binmask), "The binary mask does not exist"
    # Load the binary mask
    binmask_nifti = nib.load(path_sct_binmask)
    binmask = binmask_nifti.get_fdata()

    #Verify that the gaussian mask exists
    assert os.path.exists(path_sct_gaussmask), "The gaussian mask does not exist"
    # Load the gaussian mask
    gaussmask_nifti = nib.load(path_sct_gaussmask)
    gaussmask = gaussmask_nifti.get_fdata()

    # Verify that the output folder exists
    assert os.path.exists(os.path.dirname(path_sct_softmask)), "The output folder does not exist"
    # Create and load the gaussian soft mask
    g_softmask = gaussian_sct_softmask(path_sct_binmask, path_sct_gaussmask)
    softmask_nifti = save_softmask(g_softmask, path_sct_softmask, path_sct_binmask)
    softmask = softmask_nifti.get_fdata()

    # Verify that the soft mask has been created
    assert os.path.exists(path_sct_softmask), "The soft mask has not been created"
    # Verify that the soft mask has the correct dimensions and values
    assert softmask.shape == binmask.shape and softmask.shape == gaussmask.shape, "The soft mask has incorrect dimensions"
    assert np.array_equal(binmask_nifti.affine, softmask_nifti.affine) and np.array_equal(gaussmask_nifti.affine, softmask_nifti.affine), "The affine matrices do not match."
    assert 0.0 <= np.min(softmask) and np.max(softmask) <= 1.0, "The soft mask values are out of range"
    assert np.array_equal((softmask == 1.0), binmask.astype(bool)), "Mismatch in binary regions between binmask and softmask"
