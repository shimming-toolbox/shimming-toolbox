#!/usr/bin/env python3

import pathlib
import tempfile
import os

from click.testing import CliRunner
from shimmingtoolbox.cli.get_centerline import get_centerline_cli
from shimmingtoolbox import __dir_testing__


def test_cli_get_centerline():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        out = os.path.join(tmp, 'mask')
        method = 'fitseg'
        centerline_algo = 'linear'

        result = runner.invoke(get_centerline_cli, ['-input', inp, '-method', method, '-centerline-algo',
                                                    centerline_algo, '-output', out])

        assert result.exit_code == 0
        assert result is not None
