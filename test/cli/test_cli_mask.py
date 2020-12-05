#!/usr/bin/env python3

import numpy as np
import pathlib
import tempfile
import os
import nibabel as nib

from click.testing import CliRunner
from shimmingtoolbox.cli.mask import mask_cli
from shimmingtoolbox import __dir_testing__


def test_cli_mask_box():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out = os.path.join(tmp, 'mask.nii.gz')
        size1 = 10
        size2 = 20
        size3 = 5
        result = runner.invoke(mask_cli, ['box', '-input', inp, '-output', out, '-size', size1, size2, size3])

        # The center of the mask is the middle of the array [64, 38, 5] so the expected mask for the positions [58:62,
        # 28:31, 7:9] is :
        expected = np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                             [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                             [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                             [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

        nii = nib.load(out)
        mask = nii.get_fdata()

        assert result.exit_code == 0
        assert result is not None
        assert np.all(mask[58:62, 28:31, 7:9] == expected)


def test_cli_mask_rect():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out = os.path.join(tmp, 'mask.nii.gz')
        size1 = 10
        size2 = 20

        result = runner.invoke(mask_cli, ['rect', '-input', inp, '-output', out, '-size', size1, size2])

        # Knowing that the array is in 3 dimensions, the rectangle mask will be applied on the whole 3rd dimension. The
        # center of the rectangle mask on each slice is the middle of the 2D array [64, 38] so the expected mask for the
        # positions [58:62, 28:31, 7:9] is :
        expected = np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                             [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                             [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                             [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])

        nii = nib.load(out)
        mask = nii.get_fdata()

        assert result.exit_code == 0
        assert result is not None
        assert np.all(mask[58:62, 28:31, 7:9] == expected)


def test_cli_mask_threshold():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out = os.path.join(tmp, 'mask.nii.gz')
        thr = 780
        result = runner.invoke(mask_cli, ['threshold', '-input', inp, '-output', out, '-thr', thr])

        # With a threshold value of 780, the expected mask for the positions [58:62, 28:31, 7:9] is :
        expected = np.array([[[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                             [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                             [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                             [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])

        nii = nib.load(out)
        mask = nii.get_fdata()

        assert result.exit_code == 0
        assert result is not None
        assert np.all(mask[58:62, 28:31, 7:9] == expected)


def test_cli_mask_sct():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out = os.path.join(tmp, 'mask.nii.gz')
        process1 = 'coord'
        process2 = '20x15'
        result = runner.invoke(mask_cli, ['sct', '-input', inp, '-output', out, '-process1', process1, '-process2',
                                          process2])

        assert result.exit_code == 0
        assert result is not None