#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile

from click.testing import CliRunner
from shimmingtoolbox.cli.dicom_to_nifti import dicom_to_nifti_cli
from shimmingtoolbox import __dir_testing__


def test_cli_dicom_to_nifti():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        path_dicoms = os.path.join(__dir_testing__, 'dicom_unsorted')
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
        result = runner.invoke(dicom_to_nifti_cli, ['-input', path_dicoms, '-output', path_nifti,
                                                    '-subject', subject_id])

        assert result.exit_code == 0
        # Check that dicom_to_nifti was invoked, not if files were actually created (API test already looks for that)
        assert os.path.isdir(path_nifti)
