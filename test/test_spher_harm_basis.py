#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import os
import pytest

from shimmingtoolbox.coils.spher_harm_basis import siemens_basis, ge_basis, get_flip_matrix
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox import __dir_testing__

dummy_data = [
    np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij'),
]


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_normal_siemens_basis(x, y, z):
    basis = siemens_basis(x, y, z)

    # Test for shape
    assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))

    # Test for a value, arbitrarily chose basis[0, 0, 0, 0].
    # The full matrix could be checked to be more thorough but would require explicitly defining the matrix which is
    # 2x2x2x8.
    assert (np.allclose(basis[2, 2, 0, :], [-4.25774785e-02, 4.25774785e-02, 4.25774785e-02, -7.09057455e-21,
                                            -8.51549570e-05, 8.51549570e-05, 5.21423728e-21, -8.51549570e-05],
                        rtol=1e-09))


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_siemens_basis(x, y, z):
    basis = siemens_basis(x, y, z, orders=(1,))
    print(basis.shape)
    assert np.all(basis.shape == (3, 3, 3, 3))


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_create_scanner_coil_order3(x, y, z):
    with pytest.raises(NotImplementedError, match="Spherical harmonics not implemented for order 3 and up"):
        siemens_basis(x, y, z, orders=(3,))


def test_siemens_basis_resample():
    """
    Output spherical harmonics in a discrete space corresponding to an image
    """
    fname = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    nii = nib.load(fname)
    # affine = np.linalg.inv(nib.affine)
    affine = nii.affine

    # generate meshgrid with physical coordinates associated with the fieldmap
    coord_phys = generate_meshgrid(nii.shape, affine)

    # create SH basis in the voxel coordinate
    basis = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

    # Hard-coded values corresponding to the mid-point of the FOV.
    expected = np.array([-5.32009578e-18, 8.68837575e-02, 1.03216326e+00, 2.49330547e-02,
                         -2.57939530e-19, 4.21247220e-03, -1.77295312e-04, -2.17124136e-20])

    nx, ny, nz = nii.get_fdata().shape
    assert (np.all(np.isclose(basis[int(nx / 2), int(ny / 2), int(nz / 2), :], expected, rtol=1e-05)))


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_ge_basis(x, y, z):
    basis = ge_basis(x, y, z)

    # Test for shape
    assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))

    # Test for a value, arbitrarily chose basis[0, 0, 0, 0].
    # The full matrix could be checked to be more thorough but would require explicitly defining the matrix which is
    # 2x2x2x8.
    assert (np.allclose(basis[2, 2, 0, :], [-4.25774785e-02, -4.25774785e-02, 4.25774785e-02, 3.68477320e-05,
                                            -4.51742340e-05, -4.45028740e-05, 7.18998000e-09, 3.53411800e-07],
                        rtol=1e-09))


class TestGetFlipMatrix:
    def test_flip_cs(self):
        out = get_flip_matrix('RAS', xyz=True)
        assert np.all(out == [1, 1, 1])

    def test_flip_cs_lpi(self):
        out = get_flip_matrix('LPI', xyz=True)
        assert np.all(out == [-1, -1, -1])

    def test_flip_cs_order2(self):
        out = get_flip_matrix('LAI', xyz=False)
        assert np.all(out == [1, -1, -1, -1, -1, 1, 1, 1])

    def test_flip_cs_len4(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAIS')

    def test_flip_cs_lap(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAP')

    def test_flip_siemens(self):
        out = get_flip_matrix('LAI', xyz=False, manufacturer='Siemens')
        assert np.all(out == [-1, 1, -1, 1, 1, -1, 1, -1])

    def test_flip_ge(self):
        out = get_flip_matrix('LAI', xyz=False, manufacturer='GE')
        assert np.all(out == [-1,  1, -1, -1, -1,  1,  1,  1])
