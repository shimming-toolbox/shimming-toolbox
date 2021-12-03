#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import os
import pathlib
import tempfile
import pytest
import nibabel as nib
import numpy as np
from shutil import copyfile

from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli
from shimmingtoolbox import __dir_testing__

PHASE_SCALING_SIEMENS = 4095
fname_phasediff = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.nii.gz')
fname_mag_realtime = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
fname_mag_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
fname_phase1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
fname_phase2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')

@pytest.mark.prelude
def test_cli_prepare_fieldmap_1_echo():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag_realtime, '--output',
                                                      fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_2_echos():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phase1, fname_phase2, '--mag', fname_mag_fieldmap,
                                                      '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_default_output():
    runner = CliRunner()

    result = runner.invoke(prepare_fieldmap_cli, [fname_phase1, fname_phase2, '--mag', fname_mag_fieldmap],
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

        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag_realtime, '--output',
                                                      fname_output, '--gaussian-filter', 'True', '--sigma', 1],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


@pytest.mark.prelude
def test_cli_prepare_fieldmap_autoscale():
    """Tests the CLI with autoscale False"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Rescale input data to -pi to pi and save in tmp directory
        nii_phasediff = nib.load(fname_phasediff)
        phasediff = nii_phasediff.get_fdata()
        rescaled = phasediff * (2 * np.pi / PHASE_SCALING_SIEMENS) - np.pi
        nii_rescaled = nib.Nifti1Image(rescaled, nii_phasediff.affine, header=nii_phasediff.header)
        fname_rescaled = os.path.join(tmp, 'rescaled_phasediff.nii.gz')
        nib.save(nii_rescaled, fname_rescaled)

        # Copy the json file next to the new phase data (required by read_nii)
        fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.json')
        fname_json_tmp = os.path.join(tmp, 'rescaled_phasediff.json')
        copyfile(fname_json, fname_json_tmp)

        # Set up other paths
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_rescaled, '--mag', fname_mag_realtime, '--output',
                                                      fname_output, '--autoscale-phase', 'False'],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))


def test_cli_prepare_fieldmap_wrong_ext():

    runner = CliRunner()

    fname_output = 'fieldmap.txt'

    with pytest.raises(ValueError, match="Output filename must have one of the following extensions: '.nii', '.nii.gz'"):
        runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag_realtime, '--output', fname_output],
                      catch_exceptions=False)


@pytest.mark.prelude
def test_cli_prepare_fieldmap_default_fname_output():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '--mag', fname_mag_realtime, '--output', tmp],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.nii.gz'))
        assert os.path.isfile(os.path.join(tmp, 'fieldmap.json'))
