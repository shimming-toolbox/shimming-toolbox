#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import nibabel as nib
import os
import pathlib
import pytest
import tempfile

from shimmingtoolbox.cli.maths import maths_cli
from shimmingtoolbox import __dir_testing__

fname_input = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')


def test_mean():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'mean.nii.gz')
        result = runner.invoke(maths_cli, ['mean',
                                           '--input', fname_input,
                                           '--axis', '1',
                                           '--output', fname_output], catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert nib.load(fname_output).shape == (128, 20)


def test_mean_axis_out_of_bound():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'mean.nii.gz')
        with pytest.raises(IndexError, match="axis 3 is out of bounds for array of dimension 3"):
            runner.invoke(maths_cli, ['mean',
                                      '--input', fname_input,
                                      '--axis', '3',
                                      '--output', fname_output], catch_exceptions=False)
