#!usr/bin/env python3
# coding: utf-8

from click.testing import CliRunner
import os
import pathlib
import pytest
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
    with pytest.raises(RuntimeError, match=f"{path} does not contain dicom files"):
        CliRunner().invoke(sort_dicoms, ['-i', path], catch_exceptions=False)
