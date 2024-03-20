#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import os
import pathlib
import tempfile

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.cli.unwrap import unwrap_cli

fname_data = os.path.join(__dir_testing__, "ds_b0", "sub-realtime", "fmap", "sub-realtime_phasediff.nii.gz")
fname_mag = os.path.join(__dir_testing__, "ds_b0", "sub-realtime", "fmap", "sub-realtime_magnitude1.nii.gz")


def test_unwrap_cli():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_data,
                                            '--mag', fname_mag,
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'unwrapped.json'))
