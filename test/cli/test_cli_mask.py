#!/usr/bin/env python3

import pathlib
import tempfile
import os

from click.testing import CliRunner
from shimmingtoolbox.cli.mask import mask_cli
from shimmingtoolbox import __dir_testing__


def test_cli_mask_sct():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'nifti4')
        process1 = 'coord'
        process2 = '20x15'
        result = runner.invoke(mask_cli, ['sct', '-input', inp, '-output', out, '-process1', process1, '-process2', process2])

        assert result.exit_code == 0
        assert result is not None
        assert len(os.listdir(out)) != 0
