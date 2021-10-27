#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile
from click.testing import CliRunner

from shimmingtoolbox.cli.b1_shim import b1_shim_cli
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__


def test_b1_shim_cli():
    """Test CLI for performing RF shimming"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_b1_map = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'TB1map_axial.nii.gz')

        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_b1_shim')

        # Run the CLI
        runner = CliRunner()
        result = runner.invoke(b1_shim_cli, ['--b1map', fname_b1_map, '--output', path_output], catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_b1_shim_cli_args():
    """Test CLI for performing RF shimming"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_b1_map = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'TB1map_axial.nii.gz')
        fname_cp_json = os.path.join(__dir_shimmingtoolbox__, 'config', 'cp_mode.json')

        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_b1_shim')

        # Run the CLI
        runner = CliRunner()
        result = runner.invoke(b1_shim_cli, ['--b1map', fname_b1_map, '--cp', fname_cp_json, '--algo', 2, '--target',
                                             20, '--SED', 1.2, '--output', path_output],
                               catch_exceptions=True)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0

# TODO add test with mask
