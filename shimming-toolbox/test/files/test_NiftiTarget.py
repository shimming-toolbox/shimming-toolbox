import os
import pytest
import numpy as np
import nibabel as nib
import tempfile

from shimmingtoolbox.files.NiftiTarget import NiftiTarget


@pytest.fixture
def temp_nifti_file_4d():
    """Create a temporary 4D NIfTI file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple 4D array
        data = np.zeros((10, 10, 10, 10))
        nii = nib.Nifti1Image(data, affine=np.eye(4))

        # Save both .nii and .nii.gz files
        nii_path = os.path.join(tmpdir, "test.nii")
        nib.save(nii, nii_path)

        # Create corresponding JSON file
        json_path = os.path.join(tmpdir, "test.json")
        with open(json_path, 'w') as f:
            f.write('{"test": "data"}')

        yield nii_path


@pytest.fixture
def temp_nifti_file_3d():
    """Create a temporary 3D NIfTI file for testing."""
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


def test_niftitarget_init_4d(temp_nifti_file_4d):
    """Test NiftiTarget initialization with a 4D file."""
    nifti = NiftiTarget(temp_nifti_file_4d)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)


def test_niftitarget_init_3d(temp_nifti_file_3d):
    """Test NiftiTarget initialization with a 3D file."""
    nifti = NiftiTarget(temp_nifti_file_3d)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)


def test_set_nii_wrong_dim(temp_nifti_file_4d):
    """Test setting a new NIfTI image with wrong dimensions."""
    nifti = NiftiTarget(temp_nifti_file_4d)
    new_data = np.ones((10, 10, 10, 10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))

    with pytest.raises(ValueError, match="Target image must be in 3d or 4d"):
        nifti.set_nii(new_nii)


def test_set_nii_3d(temp_nifti_file_4d):
    """Test setting a new 3D NIfTI image."""
    nifti = NiftiTarget(temp_nifti_file_4d)
    new_data = np.ones((5, 5, 5))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    nifti.set_nii(new_nii)
    assert nifti.data.shape == (5, 5, 5)


def test_make_3d(temp_nifti_file_4d):
    """Test the make_3d method."""
    data = np.random.rand(10, 10, 10, 5)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, temp_nifti_file_4d)
    nifti = NiftiTarget(temp_nifti_file_4d)
    assert nifti.data.shape == (10, 10, 10)
    np.testing.assert_allclose(nifti.data, np.mean(data, axis=3))


def test_check_dimensions_wrong_slice_encode(temp_nifti_file_3d):
    """Test check_dimensions with wrong slice encoding direction."""
    nii = nib.load(temp_nifti_file_3d)
    nii.header.set_dim_info(freq=0, phase=1, slice=0)
    nib.save(nii, temp_nifti_file_3d)
    with pytest.raises(RuntimeError, match="Slice encode direction must be the 3rd dimension of the NIfTI file."):
        NiftiTarget(temp_nifti_file_3d)


def test_get_fat_sat_option(temp_nifti_file_4d):
    """Test getting the fat saturation option."""
    # Test with FS option
    json_data = {"ScanOptions": "FS"}
    nifti = NiftiTarget(temp_nifti_file_4d, json=json_data)
    assert nifti.get_fat_sat_option() is True

    # Test without FS option
    json_data = {"ScanOptions": "OTHER"}
    nifti = NiftiTarget(temp_nifti_file_4d, json=json_data)
    assert nifti.get_fat_sat_option() is False

    # Test without ScanOptions
    json_data = {}
    nifti = NiftiTarget(temp_nifti_file_4d, json=json_data)
    assert nifti.get_fat_sat_option() is False
