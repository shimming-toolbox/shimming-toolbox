import os
import pytest
import numpy as np
import nibabel as nib
import tempfile

from shimmingtoolbox.files.NiftiFieldMap import NiftiFieldMap

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

def test_niftifieldmap_init(temp_nifti_file):
    """Test NiftiFieldMap initialization with valid file."""
    nifti = NiftiFieldMap(temp_nifti_file, 3)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)

def test_set_nii_wrong_dim(temp_nifti_file):
    """Test setting a new NIfTI image."""
    nifti = NiftiFieldMap(temp_nifti_file, 3)
    new_data = np.ones((10, 10, 10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    
    with pytest.raises(ValueError, match="Fieldmap must be 2d or 3d"):
        nifti.set_nii(new_nii)

def test_set_nii_wrong_dim_rt_init(temp_nifti_file):
    """Test initialization with wrong dimensions for realtime processing."""
    with pytest.raises(ValueError, match="Fieldmap must be 4d for realtime processing"):
        nifti = NiftiFieldMap(temp_nifti_file, 3, is_realtime=True)

def test_set_nii_wrong_dim_rt_set(temp_nifti_file):
    """Test setting a new NIfTI image with wrong dimensions for realtime processing."""
    # Create a 4D NIfTI first for proper initialization
    data_4d = np.zeros((10, 10, 10, 5))  # 4D data
    nii_4d = nib.Nifti1Image(data_4d, affine=np.eye(4))
    nib.save(nii_4d, temp_nifti_file)
    
    nifti = NiftiFieldMap(temp_nifti_file, 3, is_realtime=True)
    new_data = np.ones((10, 10, 10))  # 3D data
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    
    with pytest.raises(ValueError, match="Fieldmap must be 4d for realtime processing"):
        nifti.set_nii(new_nii)

def test_set_nii_2d(temp_nifti_file):
    """Test setting a 2D NIfTI image."""
    nifti = NiftiFieldMap(temp_nifti_file, 3)
    new_data = np.ones((10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    nifti.set_nii(new_nii)
    assert nifti.data.shape == (10, 10, 1)

def test_extend_field_map(temp_nifti_file):
    """Test extending the field map."""
    nifti = NiftiFieldMap(temp_nifti_file, 12)
    assert nifti.extended_data.shape == (12, 12, 12)

def test_extend_field_map_realtime(temp_nifti_file):
    """Test extending the field map in real-time mode."""
    data_4d = np.zeros((10, 10, 10, 5))
    nii_4d = nib.Nifti1Image(data_4d, affine=np.eye(4))
    nib.save(nii_4d, temp_nifti_file)
    nifti = NiftiFieldMap(temp_nifti_file, 12, is_realtime=True)
    assert nifti.extended_data.shape == (12, 12, 12, 5)