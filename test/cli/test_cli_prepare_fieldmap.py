#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import os
import pathlib
import tempfile
import pytest

from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli
from shimmingtoolbox import __dir_testing__


@pytest.mark.prelude
def test_cli_prepare_fieldmap_1_echo():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_phasediff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                       'sub-example_phasediff.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag, '--output', fname_output],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_2_echos():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_phase1 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        fname_phase2 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        fname_mag = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phase1, fname_phase2, '--mag', fname_mag,
                                                      '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_default_output():
    runner = CliRunner()

    fname_phase1 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
    fname_phase2 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')

    fname_mag = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')

    result = runner.invoke(prepare_fieldmap_cli, [fname_phase1, fname_phase2, '--mag', fname_mag],
                           catch_exceptions=False)

    assert result.exit_code == 0
    assert os.path.isfile(os.path.join(os.curdir, 'fieldmap.nii.gz'))
    assert os.path.isfile(os.path.join(os.curdir, 'fieldmap.json'))
    os.remove(os.path.join(os.curdir, 'fieldmap.nii.gz'))
    os.remove(os.path.join(os.curdir, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_gaussian():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_phasediff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                       'sub-example_phasediff.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag, '--output', fname_output,
                                                      '--gaussian-filter', 'True', '--sigma', 1],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))
