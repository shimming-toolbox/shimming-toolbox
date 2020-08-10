# coding: utf-8

import os
import pathlib
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__


def test_dicom_to_nifti_unsorted():
    # TODO: put as decorator
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        # Conversion of the B0 mapping dicoms
        dicom_to_nifti(os.path.join(__dir_testing__, 'dicom_unsorted'), path_nifti, subject_id=subject_id)

        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap',
                                                       subject_id + f'_{modality}{i}.{ext}'))


def test_dicom_to_nifti_b1map():
    """"
    Test special cases of DICOM files that don't have enough information to be properly converted by dcm2niix.
    More details at: https://github.com/shimming-toolbox/shimming-toolbox-py/issues/58
    """
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'

        # Conversion of the RF maps
        dicom_to_nifti(os.path.join(__dir_testing__, 'b1_maps'), path_nifti, subject_id=subject_id, special_dicom=['TB1map'])
        for modality in ['phase', 'mag']:
            for ext in ['nii.gz', 'json']:
                assert os.path.exists(os.path.join(path_nifti, subject_id, 'rfmap',
                                                   subject_id + f'_rfmap_{modality}.{ext}'))
