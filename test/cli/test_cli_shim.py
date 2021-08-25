#!/usr/bin/python3
# -*- coding: utf-8 -*
import pytest
from click.testing import CliRunner
import tempfile
import pathlib
import os

from shimmingtoolbox.cli.shim import define_slices_cli
from shimmingtoolbox import __dir_testing__


def test_cli_define_slices_def():
    """Test using a number for the number of slices"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'slices.json')
        runner.invoke(define_slices_cli, ['--slices', '12',
                                          '--factor', '5',
                                          '--method', 'sequential',
                                          '-o', fname_output],
                      catch_exceptions=False)
        assert os.path.isfile(fname_output)


def test_cli_define_slices_anat():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                                  'sub-example_unshimmed_e1.nii.gz')
        fname_output = os.path.join(tmp, 'slices.json')
        runner.invoke(define_slices_cli, ['--slices', fname_anat,
                                          '--factor', '5',
                                          '--method', 'sequential',
                                          '-o', fname_output],
                      catch_exceptions=False)
        assert os.path.isfile(fname_output)


def test_cli_define_slices_wrong_input():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_anat = os.path.join('abc.nii')
        fname_output = os.path.join(tmp, 'slices.json')
        with pytest.raises(ValueError, match="Could not get the number of slices"):
            runner.invoke(define_slices_cli, ['--slices', fname_anat,
                                              '--factor', '5',
                                              '--method', 'sequential',
                                              '-o', fname_output],
                          catch_exceptions=False)


def test_cli_define_slices_wrong_output():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'slices')
        with pytest.raises(ValueError, match="Filename of the output must be a json file"):
            runner.invoke(define_slices_cli, ['--slices', "10",
                                              '--factor', '5',
                                              '--method', 'sequential',
                                              '-o', fname_output],
                          catch_exceptions=False)
