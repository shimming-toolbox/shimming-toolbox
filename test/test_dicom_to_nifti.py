# coding: utf-8

import os
import pathlib
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__


def test_dicom_to_nifti():
    # TODO: put as decorator
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(__dir_testing__, path_nifti, subject_id=subject_id)
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap', subject_id + '_{}{}.{}'.format(
                        modality, i, ext)))


def test_dicom_to_nifti_remove_tmp():
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(os.path.join(__dir_testing__, 'dicom_unsorted'), path_nifti, subject_id=subject_id, remove_tmp=True)
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        assert os.path.exists(path_nifti)
        assert not os.path.exists(os.path.join(path_nifti, 'tmp_dcm2bids'))
