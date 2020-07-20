# coding: utf-8

import os
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__


def test_dicom_to_nifti():
    # Create temporary folder for processing
    with tempfile.TemporaryDirectory(prefix='st_test_dicom_to_nifti_') as tmp:
        nifti_path = os.path.join(tmp, 'niftis')
        dicom_to_nifti(__dir_testing__, nifti_path)
        # Check if one of the file is created
        assert os.path.exists(os.path.join(nifti_path, 'sub-', '6_a_gre_DYNshim.json', 'sub-_run-01_MR.nii.gz'))
        # TODO: check if json is created, or maybe implement more sensitive test (e.g. integrity, etc.)
