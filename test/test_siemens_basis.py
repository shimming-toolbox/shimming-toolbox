#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import numpy as np
import pytest
import math

import nibabel

from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox import __dir_testing__

dummy_data = [
    np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij'),
]


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_normal_siemens_basis(x, y, z):

    basis = siemens_basis(x, y, z)

    # Test for shape
    assert(np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))

    # Test for a value, arbitrarily chose basis[0, 0, 0, 0].
    # The full matrix could be checked to be more thorough but would require explicitly defining the matrix which is
    # 2x2x2x8. -4.25760000e-02 was worked out to be the value that should be in basis[0, 0, 0, 0].
    assert(math.isclose(basis[0, 0, 0, 0], -4.25760000e-02, rel_tol=1e-09))


def test_siemens_basis_resample():
    """
    Output spherical harmonics in a discrete space corresponding to an image
    :return:
    """
    fname = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    nib = nibabel.load(fname)
    # affine = np.linalg.inv(nib.affine)
    affine = nib.affine
    nx, ny, nz = nib.get_fdata().shape
    coord_vox = np.meshgrid(np.array(range(nx)), np.array(range(ny)), np.array(range(nz)), indexing='ij')
    coord_phys = [np.zeros_like(coord_vox[0]), np.zeros_like(coord_vox[1]), np.zeros_like(coord_vox[2])]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                coord_phys_list = np.dot([coord_vox[i][ix, iy, iz] for i in range(3)], affine[0:3, 0:3]) + affine[0:3, 3]
                for i in range(3):
                    coord_phys[i][ix, iy, iz] = coord_phys_list[i]

    # coord_phys was checked and has the correct scanner coordinates
    # TODO: Better code ^

    basis = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

    expected = np.array([5.21405621e-18, -8.51520000e-02,  1.02182400e+00,  2.44386240e-02,
                         2.50274698e-19, -4.08729600e-03, -1.70304000e-04, -2.08562248e-20])
    assert(np.all(np.isclose(basis[int(nx/2), int(ny/2), int(nz/2), :], expected, rtol=1e-05)))

    # nibabel.save(nibabel.Nifti1Image(basis, affine), 'foo.nii.gz')
