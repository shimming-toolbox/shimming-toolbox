#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import os
import pathlib
import tempfile
from click.testing import CliRunner

from shimmingtoolbox.cli.maths import maths_cli
from shimmingtoolbox import __dir_testing__


def test_mean():
    fname_input = os.path.join(__dir_testing__,
                               'realtime_zshimming_data',
                               'nifti',
                               'sub-example',
                               'anat',
                               'sub-example_unshimmed_e1.nii.gz')

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
    fname_input = os.path.join(__dir_testing__,
                               'realtime_zshimming_data',
                               'nifti',
                               'sub-example',
                               'anat',
                               'sub-example_unshimmed_e1.nii.gz')

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'mean.nii.gz')
        try:
            runner.invoke(maths_cli, ['mean',
                                      '--input', fname_input,
                                      '--axis', '3',
                                      '--output', fname_output], catch_exceptions=False)
        except IndexError:
            # Expected behaviour
            return 0

        # Non expected behaviour
        assert False
