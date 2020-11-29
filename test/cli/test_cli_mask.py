#!/usr/bin/env python3

import pathlib
import tempfile
import os

from click.testing import CliRunner
from shimmingtoolbox.cli.mask import mask_cli
from shimmingtoolbox import __dir_testing__


def test_cli_mask_box():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti1')
        size1 = 10
        size2 = 20
        size3 = 5
        result = runner.invoke(mask_cli, ['box', '-input', inp, '-output', out, '-size', size1, size2, size3])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0


def test_cli_mask_rect():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti2')
        size1 = 50
        size2 = 5

        result = runner.invoke(mask_cli, ['rect', '-input', inp, '-output', out, '-size', size1, size2])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0


def test_cli_mask_threshold():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti3')
        thr = 30
        result = runner.invoke(mask_cli, ['threshold', '-input', inp, '-output', out, '-thr', thr])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0
