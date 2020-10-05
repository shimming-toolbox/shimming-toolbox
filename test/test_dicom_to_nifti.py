# coding: utf-8

import os
import pathlib
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__


def test_dicom_to_nifti():
    # TODO: put as decorator
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(os.path.join(__dir_shimmingtoolbox__, __dir_testing__, 'dicom_unsorted'), path_nifti, subject_id=subject_id)
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap', subject_id + '_{}{}.{}'.format(
                        modality, i, ext)))


def test_dicom_to_nifti_realtime_zshim():
    # TODO: put as decorator
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        dicom_to_nifti(os.path.join(__dir_shimmingtoolbox__, __dir_testing__, 'realtime_zshimming_data'), path_nifti,
                       subject_id=subject_id)
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.

        for sequence_type in ['anat', 'fmap', 'func']:
            if sequence_type == 'fmap':
                pass
            if sequence_type == 'anat':
                pass
            else:  # sequence_type == func
                pass

        assert True

        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap', subject_id + '_{}{}.{}'.format(
                        modality, i, ext)))