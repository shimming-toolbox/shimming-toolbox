#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import pytest
import numpy as np
import nibabel as nib
import tempfile

from shimmingtoolbox.files.NiftiMask import NiftiMask
from shimmingtoolbox.files.NiftiTarget import NiftiTarget


@pytest.fixture
def temp_nifti_file():
    """Create a temporary NIfTI file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple 3D array
        data = np.zeros((10, 10, 10))
        nii = nib.Nifti1Image(data, affine=np.eye(4))

        # Save both .nii and .nii.gz files
        nii_path = os.path.join(tmpdir, "test.nii")
        nib.save(nii, nii_path)

        # Create corresponding JSON file
        json_path = os.path.join(tmpdir, "test.json")
        with open(json_path, 'w') as f:
            f.write('{"test": "data"}')

        yield nii_path


def create_nifti_target(tmpdir, shape, affine):
    """Helper function to create a NiftiTarget for testing."""
    target_data = np.zeros(shape)
    target_nii = nib.Nifti1Image(target_data, affine)
    target_path = os.path.join(tmpdir, "target.nii")
    nib.save(target_nii, target_path)

    json_path = os.path.join(tmpdir, "target.json")
    with open(json_path, 'w') as f:
        f.write('{"test": "data"}')

    return NiftiTarget(target_path)


def test_niftimask_init(temp_nifti_file):
    """Test NiftiMask initialization with valid file."""
    nifti = NiftiMask(temp_nifti_file)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)


def test_set_nii_wrong_dim(temp_nifti_file):
    """Test setting a new NIfTI image with wrong dimensions."""
    nifti = NiftiMask(temp_nifti_file)
    new_data = np.ones((10, 10, 10, 10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))

    with pytest.raises(ValueError, match="Mask must be in 3d or 4d"):
        nifti.set_nii(new_nii, None)  # NiftiTarget is not provided here


def test_set_nii_2d(temp_nifti_file):
    """Test setting a 2D NIfTI image."""
    nifti = NiftiMask(temp_nifti_file)
    new_data = np.ones((10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))

    with pytest.raises(ValueError, match="Mask must be in 3d or 4d"):
        nifti.set_nii(new_nii, None)


def test_set_nii_4d(temp_nifti_file):
    """Test setting a 4D NIfTI image."""
    nif = NiftiMask(temp_nifti_file)
    new_data = np.zeros((10, 10, 10, 5))
    new_data[5, 5, 5, :4] = 1  # Make first 4 volumes contain the mask at a single point
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_target = create_nifti_target(tmpdir, (10, 10, 10), np.eye(4))
        nif.set_nii(new_nii, nifti_target)
        assert nif.data.shape == (10, 10, 10)
        assert np.sum(nif.data) == 0
        del nifti_target


def test_set_nii_resample(temp_nifti_file):
    """Test resampling when setting a new NIfTI image."""
    nifti = NiftiMask(temp_nifti_file)
    new_data = np.ones((5, 5, 5))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_target = create_nifti_target(tmpdir, (10, 10, 10), np.eye(4))
        nifti.set_nii(new_nii, nifti_target)
        assert nifti.data.shape == (10, 10, 10)
        del nifti_target


def test_set_nii_resample_affine(temp_nifti_file):
    """Test resampling when setting a new NIfTI image with different affine."""
    nifti = NiftiMask(temp_nifti_file)
    new_data = np.ones((10, 10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4) * 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_target = create_nifti_target(tmpdir, (10, 10, 10), np.eye(4))
        nifti.set_nii(new_nii, nifti_target)
        assert nifti.data.shape == (10, 10, 10)
        assert not np.all(nifti.affine == new_nii.affine)
        del nifti_target
