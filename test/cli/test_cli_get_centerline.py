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

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out1 = os.path.join(tmp, 'centerline1')
        method = 'fitseg'
        centerline_algo = 'linear'

        result1 = runner.invoke(get_centerline_cli, ['-input', inp, '-method', method, '-centerline_algo',
                                                     centerline_algo, '-output', out1])

        out2 = os.path.join(tmp, 'centerline2')
        method = 'fitseg'
        centerline_algo = 'nurbs'

        result2 = runner.invoke(get_centerline_cli, ['-input', inp, '-method', method, '-centerline_algo',
                                                     centerline_algo, '-output', out2])

        out3 = os.path.join(tmp, 'centerline3')
        method = 'optic'

        result3 = runner.invoke(get_centerline_cli, ['-input', inp, '-method', method, '-output', out3])

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result3.exit_code == 0
