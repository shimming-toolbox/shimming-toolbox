#!/usr/bin/python3
# -*- coding: utf-8 -*
import pytest
from click.testing import CliRunner
import tempfile
import pathlib
import os
import nibabel as nib
import numpy as np
from shutil import copy

from shimmingtoolbox.cli.shim import define_slices_cli
from shimmingtoolbox.cli.shim import shim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_config_scanner_constraints__

DEBUG = True


def _define_inputs(fmap_dim):
    # fname for fmap
    fname_fmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_fieldmap.nii.gz')
    nii = nib.load(fname_fmap)

    if fmap_dim == 4:
        nii_fmap = nii
    elif fmap_dim == 3:
        nii_fmap = nib.Nifti1Image(np.mean(nii.get_fdata(), axis=3), nii.affine, header=nii.header)
    elif fmap_dim == 2:
        nii_fmap = nib.Nifti1Image(nii.get_fdata()[..., 0, 0], nii.affine, header=nii.header)
    else:
        raise ValueError("Supported Dimensions are 2, 3 or 4")

    # fname for anat
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)
    anat = nii_anat.get_fdata()

    # Set up mask: Cube
    # static
    nx, ny, nz = anat.shape
    mask = shapes(anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)

    nii_mask = nib.Nifti1Image(mask.astype(int), nii_anat.affine)

    return nii_fmap, nii_anat, nii_mask


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask", [(
        _define_inputs(fmap_dim=3)
    )]
)
class TestCliStatic(object):
    def test_cli_static_default(self, nii_fmap, nii_anat, nii_mask):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            runner.invoke(shim_cli, ['fieldmap_static',
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask', fname_mask,
                                     '--scanner-coil-order', '1',
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))

    def test_cli_static_coils(self, nii_fmap, nii_anat, nii_mask):
        """Test cli with input coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            nii_dummy_coil = nib.Nifti1Image(np.repeat(nii_fmap.get_fdata()[..., np.newaxis], 8, axis=3),
                                             nii_fmap.affine, header=nii_fmap.header)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            runner = CliRunner()
            # TODO: use actual coil files (These are just dummy files to test if the code works)
            runner.invoke(shim_cli, ['fieldmap_static',
                                     '--coil', fname_dummy_coil, __dir_config_scanner_constraints__,
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask', fname_mask,
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))

    def test_cli_static_format_chronological_coil(self, nii_fmap, nii_anat, nii_mask):
        """Test cli with scanner coil with chronological-coil oformat"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            runner.invoke(shim_cli, ['fieldmap_static',
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask', fname_mask,
                                     '--scanner-coil-order', '1',
                                     '--slice-factor', '2',
                                     '--output-format-scanner', 'chronological-coil',
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))
            #There should be 10 x 3 values

    def test_cli_static_format_chronological_ch(self, nii_fmap, nii_anat, nii_mask):
        """Test cli with scanner coil with chronological-coil oformat"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            runner.invoke(shim_cli, ['fieldmap_static',
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask', fname_mask,
                                     '--scanner-coil-order', '1',
                                     '--slice-factor', '2',
                                     '--output-format-scanner', 'chronological-ch',
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            # There should be 3 x 10 x 1 value

    def test_cli_static_format_slicewise_ch(self, nii_fmap, nii_anat, nii_mask):
        """Test cli with scanner coil with chronological-coil oformat"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            runner.invoke(shim_cli, ['fieldmap_static',
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask', fname_mask,
                                     '--scanner-coil-order', '1',
                                     '--slice-factor', '2',
                                     '--output-format-scanner', 'slicewise-ch',
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            # There should be 3 x 20 x 1 value


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask", [(
        _define_inputs(fmap_dim=4)
    )]
)
class TestCLIRealtime(object):
    def test_cli_rt_default(self, nii_fmap, nii_anat, nii_mask):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the fieldmap
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

            # Copy fieldmap json to tmp
            fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                           'sub-example_fieldmap.json')
            copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))

            runner = CliRunner()
            runner.invoke(shim_cli, ['fieldmap_realtime',
                                     '--fmap', fname_fmap,
                                     '--anat', fname_anat,
                                     '--mask-static', fname_mask,
                                     '--mask-riro', fname_mask,
                                     '--resp', fname_resp,
                                     '--scanner-coil-order', '1',
                                     '--output', tmp],
                          catch_exceptions=False)

            if DEBUG:
                nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
                nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
                nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))

            # assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))


# def test_cli_define_slices_def():
#     """Test using a number for the number of slices"""
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         runner = CliRunner()
#         fname_output = os.path.join(tmp, 'slices.json')
#         runner.invoke(define_slices_cli, ['--slices', '12',
#                                           '--factor', '5',
#                                           '--method', 'sequential',
#                                           '-o', fname_output],
#                       catch_exceptions=False)
#         assert os.path.isfile(fname_output)
#
#
# def test_cli_define_slices_anat():
#     """Test using an anatomical file"""
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         runner = CliRunner()
#         fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
#                                   'sub-example_unshimmed_e1.nii.gz')
#         fname_output = os.path.join(tmp, 'slices.json')
#         runner.invoke(define_slices_cli, ['--slices', fname_anat,
#                                           '--factor', '5',
#                                           '--method', 'sequential',
#                                           '-o', fname_output],
#                       catch_exceptions=False)
#         assert os.path.isfile(fname_output)
#
#
# def test_cli_define_slices_wrong_input():
#     """Test using an anatomical file"""
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         runner = CliRunner()
#         fname_anat = os.path.join('abc.nii')
#         fname_output = os.path.join(tmp, 'slices.json')
#         with pytest.raises(ValueError, match="Could not get the number of slices"):
#             runner.invoke(define_slices_cli, ['--slices', fname_anat,
#                                               '--factor', '5',
#                                               '--method', 'sequential',
#                                               '-o', fname_output],
#                           catch_exceptions=False)
#
#
# def test_cli_define_slices_wrong_output():
#     """Test using an anatomical file"""
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         runner = CliRunner()
#         fname_output = os.path.join(tmp, 'slices')
#         with pytest.raises(ValueError, match="Filename of the output must be a json file"):
#             runner.invoke(define_slices_cli, ['--slices', "10",
#                                               '--factor', '5',
#                                               '--method', 'sequential',
#                                               '-o', fname_output],
#                           catch_exceptions=False)
