#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile
import nibabel as nib
import numpy as np

from click.testing import CliRunner
from shimmingtoolbox.cli.realtime_shim import realtime_shim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__


def test_cli_realtime_shim():
    """Test CLI for performing realtime shimming experiments"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Local
        fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                      'sub-example_fieldmap.nii.gz')

        # Path for mag anat image
        fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                                  'sub-example_unshimmed_e1.nii.gz')
        nii_anat = nib.load(fname_anat)
        anat = nii_anat.get_fdata()

        # Set up mask
        # Cube
        # static
        nx, ny, nz = anat.shape
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=30, len_dim2=30, len_dim3=nz)

        nii_mask_static = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask_static = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask_static, fname_mask_static)

        # Riro
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=20, len_dim2=20, len_dim3=nz)

        nii_mask_riro = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask_riro = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask_riro, fname_mask_riro)

        # Path for resp data
        fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['-fmap', fname_fieldmap,
                                                    '-mask-static', fname_mask_static,
                                                    '-mask-riro', fname_mask_riro,
                                                    '-output', path_output,
                                                    '-resp', fname_resp,
                                                    '-anat', fname_anat],
                               catch_exceptions=False)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0
