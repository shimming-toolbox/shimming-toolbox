#!usr/bin/env python3
# coding: utf-8

from click.testing import CliRunner
import copy
import os
import pathlib
import pytest
import shutil
import tempfile

from shimmingtoolbox.cli.sort_dicoms import sort_dicoms
from shimmingtoolbox import __dir_testing__


def test_sort_dicoms_cli():
    path_unsorted = os.path.join(__dir_testing__, 'dicom_unsorted')

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        result = CliRunner().invoke(sort_dicoms,
                                    ['-i', path_unsorted,
                                     '-o', tmp], catch_exceptions=True)

        assert result.exit_code == 0
        outputs = os.listdir(tmp)
        assert '06-a_gre_DYNshim' in outputs
        assert '07-a_gre_DYNshim' in outputs


def test_sort_dicoms_cli_no_dicom():
    path = os.path.join(__dir_testing__)
    with pytest.raises(RuntimeError, match="does not contain dicom files"):
        CliRunner().invoke(sort_dicoms, ['-i', path], catch_exceptions=False)


def test_sort_dicoms_cli_recursive():
    path = os.path.join(__dir_testing__, 'dicom_unsorted')
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        path_subfolder = copy.deepcopy(tmp)
        for i_subfolder in range(1, 5):
            path_subfolder = os.path.join(path_subfolder, 'subfolder' + str(i_subfolder))
            os.mkdir(path_subfolder)

        shutil.copytree(path, os.path.join(path_subfolder, 'dicoms'))

        path_output = os.path.join(tmp, 'sorted')
        result = CliRunner().invoke(sort_dicoms, ['-i', tmp, '-r', '-o', path_output], catch_exceptions=False)
        assert result.exit_code == 0
        outputs = os.listdir(path_output)
        assert '06-a_gre_DYNshim' in outputs
        assert '07-a_gre_DYNshim' in outputs
