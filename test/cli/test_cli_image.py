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
    def setup_method(self):
        path_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat')
        self.list_fname = [os.path.join(path_anat, 'sub-realtime_unshimmed_e1.nii.gz'),
                           os.path.join(path_anat, 'sub-realtime_unshimmed_e2.nii.gz'),
                           os.path.join(path_anat, 'sub-realtime_unshimmed_e3.nii.gz')]

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


class TestImageLogicalAnd(object):
    def setup_method(self):
        affine_1 = np.eye(4)
        self.nii1 = nib.Nifti1Image(np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]]), affine=affine_1)
        self.nii2 = nib.Nifti1Image(np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]]), affine=affine_1)

        mask_3 = np.ones([8, 8])
        mask_3[0, 0] = 0

        affine_3 = affine_1 * 0.5
        affine_3[3, 3] = 1
        self.nii3 = nib.Nifti1Image(mask_3, affine=affine_3)

    def test_cli_logical_and_default(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_1 = os.path.join(tmp, 'and1.nii.gz')
            nib.save(self.nii1, fname_1)
            fname_2 = os.path.join(tmp, 'and2.nii.gz')
            nib.save(self.nii2, fname_2)

            fname_output = os.path.join(tmp, 'logical_and.nii.gz')
            runner = CliRunner()
            result = runner.invoke(image_cli, ['logical-and',
                                               fname_1, fname_2,
                                               '-o', fname_output],
                                   catch_exceptions=False)

            assert result.exit_code == 0
            assert np.all(np.isclose(nib.load(fname_output).get_fdata(),
                                     np.array([[[0, 0], [1, 1]], [[0, 0], [0, 0]]])))

    def test_cli_logical_and_1_file(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_1 = os.path.join(tmp, 'and1.nii.gz')
            nib.save(self.nii1, fname_1)

            fname_output = os.path.join(tmp, 'logical_and.nii.gz')
            runner = CliRunner()
            result = runner.invoke(image_cli, ['logical-and',
                                               fname_1,
                                               '-o', fname_output],
                                   catch_exceptions=False)

            assert result.exit_code == 0
            assert np.all(np.isclose(nib.load(fname_output).get_fdata(),
                                     np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])))

    def test_cli_logical_and_diff_orient(self):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_1 = os.path.join(tmp, 'and1.nii.gz')
            nib.save(self.nii1, fname_1)
            fname_3 = os.path.join(tmp, 'and3.nii.gz')
            nib.save(self.nii3, fname_3)

            fname_output = os.path.join(tmp, 'logical_and.nii.gz')
            runner = CliRunner()
            result = runner.invoke(image_cli, ['logical-and',
                                               fname_3, fname_1,
                                               '-o', fname_output],
                                   catch_exceptions=False)

            assert result.exit_code == 0
            assert np.all(np.isclose(nib.load(fname_output).get_fdata(),
                                     np.array([[[0., 0.], [1., 0.]], [[0., 0.], [0., 0.]]])))
