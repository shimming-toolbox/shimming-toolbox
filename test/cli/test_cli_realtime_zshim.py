#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile
import nibabel as nib
import numpy as np

from click.testing import CliRunner
from shimmingtoolbox.cli.realtime_zshim import realtime_zshim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__


def test_cli_realtime_zshim():
    """Test CLI for performing realtime zshimming experiments"""
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
        nx, ny, nz = anat.shape
        mask = shapes(anat, 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=30, len_dim2=30, len_dim3=nz)

        nii_mask = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask, fname_mask)

        # Path for resp data
        fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

        # Specify output for text file and figures
        fname_output = os.path.join(__dir_shimmingtoolbox__, 'test_realtime_zshim')

        # Run the CLI
        result = runner.invoke(realtime_zshim_cli, ['-fmap', fname_fieldmap,
                                                    '-mask', fname_mask,
                                                    '-output', fname_output,
                                                    '-resp', fname_resp,
                                                    '-anat', fname_anat],
                               catch_exceptions=False)

        assert result.exit_code == 0
