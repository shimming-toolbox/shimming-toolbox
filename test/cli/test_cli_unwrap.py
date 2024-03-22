#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import math
import nibabel as nib
import os
import pathlib
import pytest
import tempfile

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.cli.unwrap import unwrap_cli

fname_data = os.path.join(__dir_testing__, "ds_b0", "sub-realtime", "fmap", "sub-realtime_phasediff.nii.gz")
fname_mag = os.path.join(__dir_testing__, "ds_b0", "sub-realtime", "fmap", "sub-realtime_magnitude1.nii.gz")

H_GYROMAGNETIC_RATIO = 42.577478518e+6  # [Hz/T]


def test_cli_unwrap_range():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')
        result = runner.invoke(unwrap_cli, ['-i', fname_data,
                                            '--mag', fname_mag,
                                            '--range', '4095',
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'unwrapped.json'))


def test_cli_unwrap_hz_dte():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Scale to Hz
        dte = 0.00246
        nii = nib.load(fname_data)
        data = (nii.get_fdata() / 4096 - 0.5) / dte
        nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        # Save
        fname_fmap_hz = os.path.join(tmp, 'fieldmap_wrapped_hz.nii.gz')
        nib.save(nii, os.path.join(tmp, fname_fmap_hz))
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_fmap_hz,
                                            '--mag', fname_mag,
                                            '--unit', 'Hz',
                                            '--dte', str(dte),
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert math.isclose(nib.load(fname_output).get_fdata()[23, 28, 0, 0], 387.24819, rel_tol=1e-5)


def test_cli_unwrap_mt_dte():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Scale to Hz
        dte = 0.00246
        nii = nib.load(fname_data)
        data = (nii.get_fdata() / 4096 - 0.5) / dte / H_GYROMAGNETIC_RATIO * 1e3  # [mT]
        nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        # Save
        fname_fmap_mt = os.path.join(tmp, 'fieldmap_wrapped_mT.nii.gz')
        nib.save(nii, os.path.join(tmp, fname_fmap_mt))
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_fmap_mt,
                                            '--mag', fname_mag,
                                            '--unit', 'mT',
                                            '--dte', str(dte),
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert math.isclose(nib.load(fname_output).get_fdata()[23, 28, 0, 0], 387.24819 / H_GYROMAGNETIC_RATIO * 1e3,
                            rel_tol=1e-5)


def test_cli_unwrap_t_dte():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Scale to Hz
        dte = 0.00246
        nii = nib.load(fname_data)
        data = (nii.get_fdata() / 4096 - 0.5) / dte / H_GYROMAGNETIC_RATIO  # [T]
        nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        # Save
        fname_fmap_t = os.path.join(tmp, 'fieldmap_wrapped_T.nii.gz')
        nib.save(nii, os.path.join(tmp, fname_fmap_t))
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_fmap_t,
                                            '--mag', fname_mag,
                                            '--unit', 'T',
                                            '--dte', str(dte),
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert math.isclose(nib.load(fname_output).get_fdata()[23, 28, 0, 0], 387.24819 / H_GYROMAGNETIC_RATIO,
                            rel_tol=1e-5)


def test_cli_unwrap_g_dte():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Scale to Hz
        dte = 0.00246
        nii = nib.load(fname_data)
        data = (nii.get_fdata() / 4096 - 0.5) / dte / H_GYROMAGNETIC_RATIO * 1e4  # [G]
        nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        # Save
        fname_fmap_g = os.path.join(tmp, 'fieldmap_wrapped_G.nii.gz')
        nib.save(nii, os.path.join(tmp, fname_fmap_g))
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_fmap_g,
                                            '--mag', fname_mag,
                                            '--unit', 'g',
                                            '--dte', str(dte),
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert math.isclose(nib.load(fname_output).get_fdata()[23, 28, 0, 0], 387.24819 / H_GYROMAGNETIC_RATIO * 1e4,
                            rel_tol=1e-5)


def test_cli_unwrap_rads_dte():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Scale to Hz
        dte = 0.00246
        nii = nib.load(fname_data)
        data = (nii.get_fdata() / 4096 - 0.5) / dte * 2 * math.pi  # [G]
        nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        # Save
        fname_fmap_rads = os.path.join(tmp, 'fieldmap_wrapped_G.nii.gz')
        nib.save(nii, os.path.join(tmp, fname_fmap_rads))
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        result = runner.invoke(unwrap_cli, ['-i', fname_fmap_rads,
                                            '--mag', fname_mag,
                                            '--unit', 'rad/s',
                                            '--dte', str(dte),
                                            '--unwrapper', 'skimage',
                                            '--output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)
        assert math.isclose(nib.load(fname_output).get_fdata()[23, 28, 0, 0], 387.24819 * 2 * math.pi,
                            rel_tol=1e-5)


def test_cli_unwrap_no_scalar():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        res = runner.invoke(unwrap_cli, ['-i', fname_data,
                                         '--mag', fname_mag,
                                         '--unit', 'Hz',
                                         '--unwrapper', 'skimage',
                                         '--output', fname_output], catch_exceptions=False)
        assert res.exit_code == 2


def test_cli_unwrap_small_range():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'unwrapped.nii.gz')

        with pytest.raises(ValueError, match="The provided --range is smaller than the range of the data."):
            runner.invoke(unwrap_cli, ['-i', fname_data,
                                       '--mag', fname_mag,
                                       '--range', '1000',
                                       '--unwrapper', 'skimage',
                                       '--output', fname_output], catch_exceptions=False)
