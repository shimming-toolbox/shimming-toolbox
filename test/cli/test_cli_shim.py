#!/usr/bin/python3
# -*- coding: utf-8 -*
import pytest
from click.testing import CliRunner
import tempfile
import pathlib
import os
import nibabel as nib

from shimmingtoolbox.cli.shim import define_slices_cli
from shimmingtoolbox.cli.shim import static_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__

DEBUG = True


def test_cli_static():

    # fname for fmap
    fname_fmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_fieldmap.nii.gz')
    nii = nib.load(fname_fmap)
    nii_fmap = nib.Nifti1Image(nii.get_fdata()[..., 0], nii.affine, header=nii.header)

    # fname for anat
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)
    anat = nii_anat.get_fdata()

    # Set up mask
    # Cube
    # static
    nx, ny, nz = anat.shape
    mask = shapes(anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)

    nii_mask = nib.Nifti1Image(mask.astype(int), nii_anat.affine)

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Save the modified fieldmap (one volume)
        fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
        nib.save(nii_fmap, fname_fmap)

        # Save the mask
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask, fname_mask)

        runner = CliRunner()
        runner.invoke(static_cli, ['--fmap', fname_fmap,
                                   '--anat', fname_anat,
                                   '--mask', fname_mask,
                                   '--scanner-coil-order', '1',
                                   '--output', tmp],
                      catch_exceptions=False)

        if DEBUG:
            nib.save(nii_fmap, os.path.join(tmp, "fmap.nii.gz"))
            nib.save(nii_anat, os.path.join(tmp, "anat.nii.gz"))
            nib.save(nii_mask, os.path.join(tmp, "mask.nii.gz"))


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
