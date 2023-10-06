#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import os
import pytest

from shimmingtoolbox.coils.spher_harm_basis import siemens_basis, ge_basis, philips_basis, get_flip_matrix
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
    # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    assert np.allclose(basis[:, 1, 1, 0], [4.25774785e-02, 0, -4.25774785e-02])
    assert np.allclose(basis[1, :, 1, 1], [-4.25774785e-02, 0, 4.25774785e-02])
    assert np.allclose(basis[1, 1, :, 2], [4.25774785e-02, 0, -4.25774785e-02])
    assert np.allclose(basis[1, 1, :, 3], [4.25774785e-05, 0.00000000e+00, 4.25774785e-05])
    assert np.allclose(basis[:, 1, :, 4], np.array([[8.5154957e-05, 0.0000000e+00, -8.5154957e-05],
                                                    [-0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                                                    [-8.5154957e-05, -0.0000000e+00, 8.5154957e-05]]))
    assert np.allclose(basis[1, :, :, 5], np.array([[-8.5154957e-05, -0.0000000e+00, 8.5154957e-05],
                                                    [0.0000000e+00, -0.0000000e+00, -0.0000000e+00],
                                                    [8.5154957e-05, 0.0000000e+00, -8.5154957e-05]]))
    assert np.allclose(basis[:, :, 1, 6], np.array([[0, 4.25774785e-05, 0],
                                                    [-4.25774785e-05, 0.00000000e+00, -4.25774785e-05],
                                                    [0, 4.25774785e-05, 0]]))
    assert np.allclose(basis[:, :, 1, 7], np.array([[-8.51549570e-05, 0, 8.51549570e-05],
                                                    [0, 0, 0],
                                                    [8.51549570e-05, -0.00000000e+00, -8.51549570e-05]]))


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
    # x, y, z, xy, zy, zx, X2 - Y2, z2
    assert np.allclose(basis[:, 1, 1, 0], [0.04257748, 0, -0.04257748])
    assert np.allclose(basis[1, :, 1, 1], [0.04257748, 0, -0.04257748])
    assert np.allclose(basis[1, 1, :, 2], [0.04257748, 0, -0.04257748])
    assert np.allclose(basis[:, :, 1, 3], np.array([[3.67369865e-05, -1.25367500e-08, -3.67310135e-05],
                                                    [1.55232500e-08, 0.00000000e+00, 1.55232500e-08],
                                                    [-3.67310135e-05, -1.25367500e-08, 3.67369865e-05]]))
    assert np.allclose(basis[1, :, :, 4], np.array([[4.47332985e-05, -4.12385000e-08, -4.49987015e-05],
                                                    [-9.14630000e-08, 0.00000000e+00, -9.14630000e-08],
                                                    [-4.49987015e-05, -4.12385000e-08, 4.47332985e-05]]))
    assert np.allclose(basis[:, 1, :, 5], np.array([[4.4341107e-05, -1.4068300e-07, -4.4354893e-05],
                                                    [1.3379000e-07, 0.0000000e+00, 1.3379000e-07],
                                                    [-4.4354893e-05, -1.4068300e-07, 4.4341107e-05]]))
    assert np.allclose(basis[:, :, 1, 6], np.array([[8.1380980e-08, 9.0716095e-06, 7.8257020e-08],
                                                    [-8.9917905e-06, 0.0000000e+00, -8.9917905e-06],
                                                    [7.8257020e-08, 9.0716095e-06, 8.1380980e-08]]))
    assert np.allclose(basis[1, 1, :, 7], [2.0056e-05, 0.0000e+00, 2.0056e-05])


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
        out = get_flip_matrix('LPI', xyz=False, manufacturer='GE')
        assert np.all(out == [-1, -1, -1, 1, 1, 1, 1, 1])

    def test_flip_philips(self):
        out = get_flip_matrix('RPI', xyz=False, manufacturer='PHILIPS')
        assert np.all(out == [1, -1, -1, -1, -1, 1, 1, 1])


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_philips_basis(x, y, z):
    # Off center z axis
    basis = philips_basis(x, y, z + 50)
    # Test for shape
    assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))
    # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    # Philips X axis is AP while Y axis is RL, therefore we check axis 1 for channel 0 and axis 0 for channel 1
    assert np.allclose(basis[1, :, 1, 0], [-42.57748, 0, 42.57748])
    assert np.allclose(basis[:, 1, 1, 1], [42.57748, 0, -42.57748])
    assert np.allclose(basis[1, 1, :, 2], [42.57748, 0, -42.57748])
    assert np.allclose(basis[1, 1, :, 3], [4.25774785e-02, 0, 4.25774785e-02])
    assert np.allclose(basis[1, :, :, 4], np.array([[-8.5154957e-02, 0, 8.5154957e-02],
                                                    [0, 0, 0],
                                                    [8.5154957e-02, 0, -8.5154957e-02]]))
    assert np.allclose(basis[:, 1, :, 5], np.array([[8.5154957e-02, 0, -8.5154957e-02],
                                                    [0, 0, 0],
                                                    [-8.5154957e-02, 0, 8.5154957e-02]]))
    assert np.allclose(basis[:, :, 1, 6], np.array([[0, -4.25774785e-02, 0],
                                                    [4.25774785e-02, 0, 4.25774785e-02],
                                                    [0, -4.25774785e-02, 0]]))
    assert np.allclose(basis[:, :, 1, 7], np.array([[-8.51549570e-02, 0, 8.51549570e-02],
                                                    [0, 0, 0],
                                                    [8.51549570e-02, 0, -8.51549570e-02]]))
