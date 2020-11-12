#!/usr/bin/env python3

import pathlib
import tempfile
import os

from click.testing import CliRunner
from shimmingtoolbox.cli.mask import mask
from shimmingtoolbox import __dir_testing__


def test_cli_mask_cube():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti1')
        size = 5

        result = runner.invoke(mask, ['cube', '-input', inp, '-output', out, '-size', size])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0


def test_cli_mask_square():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti2')
        size = 50

        result = runner.invoke(mask, ['square', '-input', inp, '-output', out, '-size', size])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0


def test_cli_mask_threshold():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti3')
        thr = 30
        result = runner.invoke(mask, ['threshold', '-input', inp, '-output', out, '-thr', thr])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0
