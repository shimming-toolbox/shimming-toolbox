#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import os
import pathlib
import tempfile
from click.testing import CliRunner

from shimmingtoolbox.cli.image import image_cli
from shimmingtoolbox import __dir_testing__


class TestImageConcat(object):
    def setup(self):
        path_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat')
        self.list_fname = [os.path.join(path_anat, 'sub-example_unshimmed_e1.nii.gz'),
                           os.path.join(path_anat, 'sub-example_unshimmed_e2.nii.gz'),
                           os.path.join(path_anat, 'sub-example_unshimmed_e3.nii.gz')]

    def test_cli_concat(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            runner = CliRunner()
            fname_output = os.path.join(tmp, 'concat.nii.gz')
            result = runner.invoke(image_cli, ['concat',
                                               self.list_fname[0],
                                               self.list_fname[1],
                                               self.list_fname[2],
                                               '--axis', '4',
                                               '--output', fname_output], catch_exceptions=False)

            assert result.exit_code == 0
            assert os.path.isfile(fname_output)
            assert nib.load(fname_output).shape == (128, 68, 20, 1, 3)

    def test_cli_concat_pixdim(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            runner = CliRunner()
            fname_output = os.path.join(tmp, 'concat.nii.gz')
            result = runner.invoke(image_cli, ['concat',
                                               self.list_fname[0],
                                               self.list_fname[1],
                                               self.list_fname[2],
                                               '--axis', '4',
                                               '--output', fname_output,
                                               '--pixdim', '0.2'], catch_exceptions=False)

            assert result.exit_code == 0
            assert os.path.isfile(fname_output)
            assert np.all(np.isclose(nib.load(fname_output).header['pixdim'],
                                     [-1, 2.1875, 2.1875, 3.6, 1, 0.2, 0, 0]))
