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
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox import __dir_testing__


def test_cli_realtime_zshim():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_fieldmap = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-example_fieldmap.nii.gz')
        nii_fmap = nib.load(fname_fieldmap)
        fmap = nii_fmap.get_fdata()
        affine = nii_fmap.affine

        # Set up mask
        nx, ny, nz, nt = fmap.shape
        mask = shapes(fmap[:, :, :, 0], 'cube', center_dim1=round(nx / 2) - 5, len_dim1=40, len_dim2=40,
                      len_dim3=nz)

        nii_mask = nib.Nifti1Image(mask.astype(int), affine)
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask, fname_mask)

        # Set up coils
        coord_phys = generate_meshgrid(fmap.shape[0:3], affine)
        coil_profile = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

        nii_coil = nib.Nifti1Image(coil_profile, affine)
        fname_coil = os.path.join(tmp, 'coil_profile.nii.gz')
        nib.save(nii_coil, fname_coil)

        result = runner.invoke(realtime_zshim, ['-fmap', fname_fieldmap, '-coil', fname_coil, '-mask', fname_mask])

        assert result.exit_code == 0
