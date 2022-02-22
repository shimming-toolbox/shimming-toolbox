#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import shutil
import tempfile

from shimmingtoolbox import __dir_testing__
from click.testing import CliRunner
from shimmingtoolbox.cli.b1shim import b1shim_cli

path_b1_folder = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'rfmap')
fname_b1_nii_axial = os.path.join(path_b1_folder, 'sub-tb1tfl_run-01_TB1map_uncombined.nii.gz')
fname_b1_nii_coronal = os.path.join(path_b1_folder, 'sub-tb1tfl_run-02_TB1map_uncombined.nii.gz')
fname_b1_nii_sagittal = os.path.join(path_b1_folder, 'sub-tb1tfl_run-03_TB1map_uncombined.nii.gz')

path_sar_file = os.path.join(__dir_testing__, 'ds_tb1', 'derivatives', 'shimming-toolbox', 'sub-tb1tfl',
                             'sub-tb1tfl_SarDataUser.mat')
fname_mask = os.path.join(__dir_testing__, 'ds_tb1', 'derivatives', 'shimming-toolbox', 'sub-tb1tfl',
                          'sub-tb1tfl_mask.nii.gz')


def test_b1shim_cli():
    """Test CLI for performing RF shimming with axial B1+ maps"""
    # Run the CLI
    result = CliRunner().invoke(b1shim_cli, ['--b1', fname_b1_nii_axial], catch_exceptions=True)
    assert len(os.listdir(os.path.join(os.curdir, 'b1_shim_results'))) != 0
    assert result.exit_code == 0
    shutil.rmtree(os.path.join(os.curdir, 'b1_shim_results'))


def test_b1shim_cli_args():
    """Test CLI for performing RF shimming with all arguments"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'b1_shim_results')

        # Run the CLI
        result = CliRunner().invoke(b1shim_cli, ['--b1', fname_b1_nii_axial, '--mask', fname_mask, '--algo', '2',
                                                 '--target', '20', '--vop', path_sar_file, '--sar_factor', '1.2',
                                                 '--output', path_output], catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_b1shim_cli_coronal():
    """Test CLI for performing RF shimming with coronal B1+ maps"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'b1_shim_results')

        # Run the CLI
        result = CliRunner().invoke(b1shim_cli, ['--b1', fname_b1_nii_coronal, '--mask', fname_mask, '--algo', '2',
                                                 '--target', '20', '--vop', path_sar_file, '--sar_factor', '1.2',
                                                 '--output', path_output], catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_b1shim_cli_sagittal():
    """Test CLI for performing RF shimming with sagittal B1+ maps"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'b1_shim_results')

        # Run the CLI
        result = CliRunner().invoke(b1shim_cli, ['--b1', fname_b1_nii_sagittal, '--mask', fname_mask, '--algo', '2',
                                                 '--target', '20', '--vop', path_sar_file, '--sar_factor', '1.2',
                                                 '--output', path_output], catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0
