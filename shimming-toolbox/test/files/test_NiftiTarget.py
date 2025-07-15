import os
import pytest
import numpy as np
import nibabel as nib
import tempfile

from shimmingtoolbox.files.NiftiTarget import NiftiTarget


@pytest.fixture
def temp_nifti_file():
    """Create a temporary NIfTI file for testing."""
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

def test_niftitarget_init(temp_nifti_file):
    """Test NiftiTarget initialization with valid file."""
    nifti = NiftiTarget(temp_nifti_file)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)
    
def test_set_nii(temp_nifti_file):
    """Test setting a new NIfTI image."""
    nifti = NiftiTarget(temp_nifti_file)
    new_data = np.ones((10, 10, 10, 10, 10))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    
    with pytest.raises(ValueError, match="Target image must be in 3d or 4d"):
        nifti.set_nii(new_nii)
