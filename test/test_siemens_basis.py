#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import numpy as np
import pytest
import math

import nibabel

from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.coils.coordinates import generate_meshgrid
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
    """
    fname = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    nii = nibabel.load(fname)
    # affine = np.linalg.inv(nib.affine)
    affine = nii.affine

    # generate meshgrid with physical coordinates associated with the fieldmap
    coord_phys = generate_meshgrid(nii.shape, affine)

    # create SH basis in the voxel coordinate
    basis = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

    # Hard-coded values corresponding to the mid-point of the FOV.
    expected = np.array([5.21405621e-18, -8.51520000e-02,  1.02182400e+00,  2.44386240e-02,
                         2.50274698e-19, -4.08729600e-03, -1.70304000e-04, -2.08562248e-20])

    nx, ny, nz = nii.get_fdata().shape
    assert(np.all(np.isclose(basis[int(nx/2), int(ny/2), int(nz/2), :], expected, rtol=1e-05)))
