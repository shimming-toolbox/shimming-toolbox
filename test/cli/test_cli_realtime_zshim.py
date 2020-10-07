#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile

from click.testing import CliRunner
from shimmingtoolbox.cli.realtime_zshim import realtime_zshim
from shimmingtoolbox import __dir_testing__


def test_cli_realtime_zshim():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_fieldmap = os.path.join(__dir_testing__, 'nifti', 'sub-example', 'fmap', 'sub-example_fieldmap.nii.gz')

        # Set up mask
        # TODO
        full_mask = shapes(fieldmaps[:, :, :, 0], 'cube', center_dim1=round(nx / 2) - 5, len_dim1=40, len_dim2=40,
                           len_dim3=nz)

        path_dicoms = os.path.join(__dir_testing__, 'dicom_unsorted')
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'

        result = runner.invoke(realtime_zshim, ['-i', path_dicoms, '-o', path_nifti, '-s', subject_id])

        assert result.exit_code == 0
        # Check that dicom_to_nifti was invoked, not if files were actually created (API test already looks for that)
        assert os.path.isdir(path_nifti)