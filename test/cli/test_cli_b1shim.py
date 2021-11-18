#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile
import shutil

from click.testing import CliRunner
from shimmingtoolbox.cli.b1shim import b1shim_cli
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__

fname_b1_nifti = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'TB1map_axial.nii.gz')
path_sar_file = os.path.join(__dir_testing__, 'b1_maps', 'vop', 'SarDataUser.mat')


def test_b1shim_cli():
    """Test CLI for performing RF shimming"""

    # Run the CLI
    runner = CliRunner()
    result = runner.invoke(b1shim_cli, ['--b1map', fname_b1_nifti], catch_exceptions=True)
    assert len(os.listdir(os.path.join(os.curdir, 'b1_shim_results'))) != 0
    assert result.exit_code == 0
    shutil.rmtree(os.path.join(os.curdir, 'b1_shim_results'))


def test_b1shim_cli_args():
    """Test CLI for performing RF shimming"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_cp_json = os.path.join(__dir_shimmingtoolbox__, 'config', 'cp_mode.json')

        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'b1_shim_results')

        # Run the CLI
        runner = CliRunner()
        result = runner.invoke(b1shim_cli, ['--b1map', fname_b1_nifti, '--cp', fname_cp_json, '--algo', 2, '--target',
                                            20, '--vop', path_sar_file, '--sed', 1.2, '--output', path_output],
                               catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0

# TODO add test with mask
