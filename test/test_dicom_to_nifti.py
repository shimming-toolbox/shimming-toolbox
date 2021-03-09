# coding: utf-8

import os
import pathlib
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__
import pytest


def test_dicom_to_nifti():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(
            path_dicom=os.path.join(__dir_testing__, 'dicom_unsorted'),
            path_nifti=path_nifti,
            subject_id=subject_id
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap', subject_id + '_{}{}.{}'.format(
                        modality, i, ext)))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_realtime_zshim(test_dcm2niix_installation):
    """Test dicom_to_nifti outputs the correct files for realtime_zshimming_data"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(
            path_dicom=os.path.join(__dir_testing__, 'realtime_zshimming_data'),
            path_nifti=path_nifti,
            subject_id=subject_id
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.

        sequence_type = 'fmap'
        for i in range(2):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    if modality == 'phase':
                        assert os.path.exists(os.path.join(path_nifti, subject_id, sequence_type,
                                                           subject_id + f"_{'phasediff'}.{ext}"))
                    else:
                        assert os.path.exists(os.path.join(path_nifti, subject_id, sequence_type,
                                                           subject_id + f"_{modality}{i+1}.{ext}"))

        sequence_type = 'anat'
        for i in range(3):
            for ext in ['nii.gz', 'json']:
                assert os.path.exists(
                    os.path.join(path_nifti, subject_id, sequence_type, subject_id + f"_unshimmed_e{i+1}.{ext}"))

        sequence_type = 'func'
        for ext in ['nii.gz', 'json']:
            assert os.path.exists(
                os.path.join(path_nifti, subject_id, sequence_type, subject_id + f"_bold.{ext}"))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_remove_tmp(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(
            path_dicom=os.path.join(__dir_testing__, 'dicom_unsorted'),
            path_nifti=path_nifti,
            subject_id=subject_id,
            remove_tmp=True
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        assert os.path.exists(path_nifti)
        assert not os.path.exists(os.path.join(path_nifti, 'tmp_dcm2bids'))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_path_dicom_invalid(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        with pytest.raises(FileNotFoundError, match=r"No dicom path found"):
            dicom_to_nifti(
                path_dicom=os.path.join(__dir_testing__, 'invalid_folder'),
                path_nifti=path_nifti,
                subject_id=subject_id
            )


@pytest.mark.dcm2niix
def test_dicom_to_nifti_path_config_invalid(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        with pytest.raises(FileNotFoundError, match=r"No dcm2bids config file found"):
            dicom_to_nifti(
                path_dicom=os.path.join(__dir_testing__, 'dicom_unsorted'),
                path_nifti=path_nifti,
                subject_id=subject_id,
                path_config_dcm2bids=os.path.join(tmp, "invalid_folder")
            )
