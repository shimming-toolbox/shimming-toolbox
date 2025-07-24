#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path
import json

from shimmingtoolbox.files.NiftiFile import NiftiFile, NIFTI_EXTENSIONS


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
        nifti = NiftiFile(temp_nifti_file, path_output=tmpdir)
        nifti.save("saved.nii.gz")
        assert os.path.exists(save_path)

        # Test default save path
        nifti_default = NiftiFile(temp_nifti_file)
        nifti_default.save()
        expected_path = os.path.join(nifti_default.path_nii, f"{nifti_default.filename}_saved.nii.gz")
        assert os.path.exists(expected_path)


def test_niftifile_save_invalid_path(temp_nifti_file):
    """Test saving with an invalid path."""
    with tempfile.NamedTemporaryFile() as tmpfile:
        with pytest.raises(ValueError):
            nifti = NiftiFile(temp_nifti_file, path_output=tmpfile.name)
            nifti.save()


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


def test_niftifile_set_nii(temp_nifti_file):
    """Test setting a new NIfTI image."""
    nifti = NiftiFile(temp_nifti_file)
    new_data = np.ones((5, 5, 5))
    new_nii = nib.Nifti1Image(new_data, affine=np.eye(4))
    nifti.set_nii(new_nii)
    assert nifti.data.shape == (5, 5, 5)
    np.testing.assert_array_equal(nifti.data, new_data)


def test_niftifile_get_json_info(temp_nifti_file, caplog):
    """Test getting info from the JSON file."""
    nifti = NiftiFile(temp_nifti_file)
    assert nifti.get_json_info("test") == "data"
    with pytest.raises(KeyError):
        nifti.get_json_info("nonexistent")
    assert nifti.get_json_info("nonexistent", required=False) is None
    assert "Key 'nonexistent' not found in JSON file" in caplog.text


@pytest.mark.parametrize("patient_position, expected_isocenter", [
    ("HFS", [-10, -20, -30]),
    ("HFP", [10, 20, -30]),
    ("FFS", [10, -20, 30]),
    ("FFP", [-10, 20, 30]),
])
def test_niftifile_get_isocenter(temp_nifti_file, patient_position, expected_isocenter):
    """Test getting the isocenter with different patient positions."""
    json_data = {
        "TablePosition": [10, 20, 30],
        "PatientPosition": patient_position
    }
    nifti = NiftiFile(temp_nifti_file, json=json_data)
    isocenter = nifti.get_isocenter()
    np.testing.assert_array_equal(isocenter, expected_isocenter)


def test_niftifile_get_frequency(temp_nifti_file):
    """Test getting the imaging frequency."""
    json_data = {"ImagingFrequency": 123.45}
    nifti = NiftiFile(temp_nifti_file, json=json_data)
    assert nifti.get_frequency() == 123450000


@pytest.mark.parametrize("shim_settings, expected_shim_settings", [
    ([1, 2, 3], {'0': [123450000], '1': [1, 2, 3], '2': None, '3': None}),
    ([1, 2, 3, 4, 5, 6, 7, 8], {'0': [123450000], '1': [1, 2, 3], '2': [4, 5, 6, 7, 8], '3': None}),
    ([1, 2], {'0': [123450000], '1': None, '2': None, '3': None}),
])
def test_niftifile_get_scanner_shim_settings(temp_nifti_file, shim_settings, expected_shim_settings):
    """Test getting scanner shim settings with different ShimSetting lengths."""
    json_data = {
        "ImagingFrequency": 123.45,
        "ShimSetting": shim_settings
    }
    nifti = NiftiFile(temp_nifti_file, json=json_data)
    retrieved_settings = nifti.get_scanner_shim_settings()
    assert retrieved_settings == expected_shim_settings


def test_niftifile_get_manufacturers_model_name(temp_nifti_file):
    """Test getting the manufacturer's model name."""
    json_data = {"ManufacturersModelName": "Test Model S"}
    nifti = NiftiFile(temp_nifti_file, json=json_data)
    assert nifti.get_manufacturers_model_name() == "Test_Model_S"


def test_niftifile_no_json(temp_nifti_file):
    """Test NiftiFile without a JSON file."""
    os.remove(os.path.join(os.path.dirname(temp_nifti_file), "test.json"))
    with pytest.raises(OSError):
        NiftiFile(temp_nifti_file)
    nifti = NiftiFile(temp_nifti_file, json_needed=False)
    assert nifti.json is None


def test_niftifile_save_json(temp_nifti_file):
    """Test saving JSON data."""
    nifti = NiftiFile(temp_nifti_file, json={"test": "data"})
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti.path_output = tmpdir
        nifti.save()
        json_path = os.path.join(tmpdir, "test_saved.json")
        assert os.path.exists(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data == {"test": "data"}

def test_niftifile_save_json_no_data(temp_nifti_file):
    """Test saving NiftiFile without JSON data."""
    nifti = NiftiFile(temp_nifti_file, json_needed=False)
    nifti.json = None
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti.path_output = tmpdir
        nifti.save()
        json_path = os.path.join(tmpdir, "test_saved.json")
        assert not os.path.exists(json_path)  # No JSON file should be created if no data is provided

def test_niftifile_save_invalid_extension(temp_nifti_file):
    """Test saving with an invalid file extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti = NiftiFile(temp_nifti_file, path_output=tmpdir)
        with pytest.raises(ValueError, match="File name must end with .nii or .nii.gz"):
            nifti.save("invalid.txt")

def test_niftifile_save_with_custom_filename(temp_nifti_file):
    """Test saving with a custom filename."""
    nifti = NiftiFile(temp_nifti_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti.path_output = tmpdir
        nifti.save("custom_name.nii.gz")
        saved_path = os.path.join(tmpdir, "custom_name.nii.gz")
        assert os.path.exists(saved_path)
        assert nib.load(saved_path).shape == (10, 10, 10)

def test_niftifile_save_with_custom_filename_no_extension(temp_nifti_file):
    """Test saving with a custom filename without extension."""
    nifti = NiftiFile(temp_nifti_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti.path_output = tmpdir
        nifti.save("custom_name")
        saved_path = os.path.join(tmpdir, "custom_name.nii.gz")
        assert os.path.exists(saved_path)
        assert nib.load(saved_path).shape == (10, 10, 10)
