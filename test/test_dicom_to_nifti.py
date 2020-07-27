# coding: utf-8
import dcm2bids
import os
from packaging import version
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__


def test_dcm2bids_versiondicom_to_nifti_version():

def test_dicom_to_nifti():
    # Create temporary folder for processing the data and execute the tests
    with tempfile.TemporaryDirectory(prefix='st_test_dicom_to_nifti_') as tmp:
        # path_nifti is the folder where the niftis will be temporarily stored
        path_nifti = os.path.join(tmp, 'niftis')
        dicom_to_nifti(__dir_testing__, path_nifti)
        # Check that all the files (.nii.gz and .json) are created with the expected names
        for i in range(1, 7):
            i = str(i)
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '6_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.nii.gz'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '6_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.json'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '7_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.nii.gz'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '7_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.json'))
