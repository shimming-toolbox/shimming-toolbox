import os
import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from shimmingtoolbox.files.file import NiftiFile, NIFTI_EXTENSIONS

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

def test_niftifile_init(temp_nifti_file):
    """Test NiftiFile initialization with valid file."""
    nifti = NiftiFile(temp_nifti_file)
    assert isinstance(nifti.nii, nib.Nifti1Image)
    assert isinstance(nifti.data, np.ndarray)
    assert nifti.data.shape == (10, 10, 10)

def test_niftifile_invalid_path():
    """Test NiftiFile initialization with invalid path."""
    with pytest.raises(ValueError):
        NiftiFile("nonexistent.nii")

def test_niftifile_invalid_extension():
    """Test NiftiFile initialization with invalid extension."""
    with pytest.raises(ValueError):
        NiftiFile("test.txt")

def test_niftifile_relative_path(temp_nifti_file):
    """Test NiftiFile with relative path."""
    current_dir = os.getcwd()
    # Change to the directory of the temporary NIfTI file
    os.chdir(os.path.dirname(temp_nifti_file))
    
    try:
        nifti = NiftiFile(os.path.basename(temp_nifti_file))
        assert os.path.isabs(nifti.path_nii)
        assert nifti.path_nii == os.getcwd()
    finally:
        os.chdir(current_dir)

def test_niftifile_load_json(temp_nifti_file):
    """Test JSON file loading."""
    nifti = NiftiFile(temp_nifti_file)
    assert nifti.json is not None
    assert nifti.json == {"test": "data"}

def test_niftifile_save(temp_nifti_file):
    """Test saving NiftiFile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test explicit save path
        save_path = os.path.join(tmpdir, "saved.nii.gz")
        nifti = NiftiFile(temp_nifti_file, path_output=save_path)
        nifti.save()
        assert os.path.exists(save_path)
        
        # Test default save path
        nifti_default = NiftiFile(temp_nifti_file)
        nifti_default.save()
        expected_path = os.path.join(nifti_default.path_nii, f"{nifti_default.filename}_saved.nii.gz")
        assert os.path.exists(expected_path)

def test_niftifile_get_filename(temp_nifti_file):
    """Test filename extraction."""
    nifti = NiftiFile(temp_nifti_file)
    assert nifti.filename == "test"

def test_niftifile_get_dirname(temp_nifti_file):
    """Test dirname extraction."""
    nifti = NiftiFile(temp_nifti_file)
    assert os.path.isabs(nifti.path_nii)
    assert os.path.exists(nifti.path_nii)

@pytest.mark.parametrize("ext", NIFTI_EXTENSIONS)
def test_niftifile_extensions(temp_nifti_file, ext):
    """Test different NIfTI extensions."""
    dirname = os.path.dirname(temp_nifti_file)
    filename = "test" + ext
    new_path = os.path.join(dirname, filename)
    
    # Convert file to required extension
    if ext != os.path.splitext(temp_nifti_file)[-1]:
        nii = nib.load(temp_nifti_file)
        nib.save(nii, new_path)
    
    nifti = NiftiFile(new_path)
    assert nifti.filename == "test"
    assert isinstance(nifti.nii, nib.Nifti1Image)