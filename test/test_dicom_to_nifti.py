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
        dicom_to_nifti(__dir_testing__, path_nifti)
        # Check that all the files (.nii.gz and .json) are created with the expected names
        for i in range(1, 7):
            i = str(i)
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '6_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.nii.gz'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '6_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.json'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '7_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.nii.gz'))
            assert os.path.exists(os.path.join(path_nifti, 'sub-', '7_a_gre_DYNshim', 'sub-_run-0' + i + '_MR.json'))
