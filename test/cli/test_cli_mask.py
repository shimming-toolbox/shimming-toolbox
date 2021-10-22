#!/usr/bin/env python3

import numpy as np
import pathlib
import tempfile
import os
import nibabel as nib
import pytest

from click.testing import CliRunner
from shimmingtoolbox.cli.mask import mask_cli
from shimmingtoolbox import __dir_testing__
# from shimmingtoolbox


def test_cli_mask_box():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        inp = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        out = os.path.join(tmp, 'mask.nii.gz')
        size1 = 10
        size2 = 20
        size3 = 5
        result = runner.invoke(mask_cli, ['box', '--input', inp, '--output', out, '--size', size1, size2, size3])

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

        result = runner.invoke(mask_cli, ['rect', '--input', inp, '--output', out, '--size', size1, size2])

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
        result = runner.invoke(mask_cli, ['threshold', '--input', inp, '--output', out, '--thr', thr])

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


@pytest.mark.sct
def test_cli_mask_sct_default():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_input = os.path.join(__dir_testing__, 't2', 't2.nii.gz')
        fname_output = os.path.join(tmp, 'mask.nii.gz')

        result = runner.invoke(mask_cli, f"sct --input {fname_input} --output {fname_output} --remove-tmp 0",
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert len(os.listdir(tmp)) == 2
        assert os.path.isfile(fname_output)


@pytest.mark.sct
def test_cli_mask_sct_all_flags():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_input = os.path.join(__dir_testing__, 't2', 't2.nii.gz')
        fname_output = os.path.join(tmp, 'mask.nii.gz')

        result = runner.invoke(mask_cli, f"sct --input {fname_input} --output {fname_output} --size 11 --shape gaussian"
                                         f" --contrast t1 --brain 0 --kernel 2d --centerline cnn"
                                         f" --verbose 1", catch_exceptions=False)

        assert result.exit_code == 0
        assert len(os.listdir(tmp)) == 1
        assert os.path.isfile(fname_output)


@pytest.mark.sct
def test_cli_mask_sct_4d():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_3d = os.path.join(__dir_testing__, 't2', 't2.nii.gz')
        nii_3d = nib.load(fname_3d)
        data_4d = np.expand_dims(nii_3d.get_fdata(), 3)
        data_4d = np.append(data_4d, data_4d, 3)
        nii_4d = nib.Nifti1Image(data_4d, nii_3d.affine, nii_3d.header)
        fname_4d = os.path.join(tmp, 't2_4d.nii.gz')
        nib.save(nii_4d, fname_4d)

        fname_output = os.path.join(tmp, 'mask.nii.gz')

        result = runner.invoke(mask_cli, f"sct --input {fname_4d} --output {fname_output} --remove-tmp 1",
                               catch_exceptions=False)

        assert result.exit_code == 0
        # There should be 2 files left since we remove tmp files: the input 4d file and the mask.
        assert len(os.listdir(tmp)) == 2
        assert os.path.isfile(fname_output)
