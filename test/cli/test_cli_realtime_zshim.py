#!usr/bin/env python3
# coding: utf-8

import os
import pathlib
import tempfile
import nibabel as nib
import numpy as np

from click.testing import CliRunner
from shimmingtoolbox.cli.realtime_zshim import realtime_zshim
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox import __dir_testing__


def test_cli_realtime_zshim():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Local
        fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                      'sub-example_fieldmap.nii.gz')
        # fname_fieldmap = os.path.join(__dir_testing__, 'test_realtime_zshim', 'sub-example_fieldmap.nii.gz')
        nii_fmap = nib.load(fname_fieldmap)

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
        # Threshold
        # mask = threshold(anat, thr=50)

        nii_mask = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask, fname_mask)

        # Set up coils
        coord_phys = generate_meshgrid(nii_fmap.get_fdata().shape[0:3], nii_fmap.affine)
        coil_profile = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

        nii_coil = nib.Nifti1Image(coil_profile, nii_fmap.affine)
        fname_coil = os.path.join(tmp, 'coil_profile.nii.gz')
        nib.save(nii_coil, fname_coil)

        # Path for resp data
        fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

        # Path for json file
        fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_magnitude1.json')

        result = runner.invoke(realtime_zshim, ['-fmap', fname_fieldmap, '-coil', fname_coil, '-mask', fname_mask,
                                                '-resp', fname_resp, '-json', fname_json, '-anat', fname_anat],
                               catch_exceptions=False)

        assert result.exit_code == 0
