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
    basis = siemens_basis(x, y, z, orders=(0, 1, 2, 3))

    # Test for shape
    assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 13)))
    # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    assert np.allclose(basis[:, 1, 1, 0], [-1, -1, -1])
    assert np.allclose(basis[:, 1, 1, 1], [4.25774785e-02, 0, -4.25774785e-02])
    assert np.allclose(basis[1, :, 1, 2], [-4.25774785e-02, 0, 4.25774785e-02])
    assert np.allclose(basis[1, 1, :, 3], [4.25774785e-02, 0, -4.25774785e-02])
    assert np.allclose(basis[1, 1, :, 4], [4.25774785e-05, 0.00000000e+00, 4.25774785e-05])
    assert np.allclose(basis[:, 1, :, 5], np.array([[8.5154957e-05, 0.0000000e+00, -8.5154957e-05],
                                                    [-0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                                                    [-8.5154957e-05, -0.0000000e+00, 8.5154957e-05]]))
    assert np.allclose(basis[1, :, :, 6], np.array([[-8.5154957e-05, -0.0000000e+00, 8.5154957e-05],
                                                    [0.0000000e+00, -0.0000000e+00, -0.0000000e+00],
                                                    [8.5154957e-05, 0.0000000e+00, -8.5154957e-05]]))
    assert np.allclose(basis[:, :, 1, 7], np.array([[0, 4.25774785e-05, 0],
                                                    [-4.25774785e-05, 0.00000000e+00, -4.25774785e-05],
                                                    [0, 4.25774785e-05, 0]]))
    assert np.allclose(basis[:, :, 1, 8], np.array([[-8.51549570e-05, 0, 8.51549570e-05],
                                                    [0, 0, 0],
                                                    [8.51549570e-05, -0.00000000e+00, -8.51549570e-05]]))
    # TODO: add tests for order 3


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_siemens_basis(x, y, z):
    basis = siemens_basis(x, y, z, orders=(1,))
    assert np.all(basis.shape == (3, 3, 3, 3))


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_create_siemens_basis_order3(x, y, z):
    basis = siemens_basis(x, y, z, orders=(3,))
    assert np.all(basis.shape == (3, 3, 3, 4))


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_create_siemens_basis_order4(x, y, z):
    with pytest.raises(NotImplementedError, match="Spherical harmonics not implemented for order 4 and up"):
        siemens_basis(x, y, z, orders=(4,))


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
    assert np.allclose(basis[:, 1, 1, 0], [0.00425775, 0, -0.00425775])
    assert np.allclose(basis[1, :, 1, 1], [0.00425775, 0, -0.00425775])
    assert np.allclose(basis[1, 1, :, 2], [0.00425775, 0, -0.00425775])
    assert np.allclose(basis[:, :, 1, 3], np.array([[-2.00681342e-05, -1.00644870e-05, -2.00438658e-05],
                                                    [-9.99151300e-06, 0.00000000e+00, -9.99151300e-06],
                                                    [-2.00438658e-05, -1.00644870e-05, -2.00681342e-05]]))
    assert np.allclose(basis[1, :, :, 4], np.array([[1.009365e-07, 3.446500e-09, 3.974650e-08],
                                                    [6.689500e-08, 0.000000e+00, 6.689500e-08],
                                                    [3.974650e-08, 3.446500e-09, 1.009365e-07]]))
    assert np.allclose(basis[:, 1, :, 5], np.array([[1.0684125e-07, 6.6350750e-08, -6.5602750e-08],
                                                    [-4.5731500e-08, 0.0000000e+00, -4.5731500e-08],
                                                    [-6.5602750e-08, 6.6350750e-08, 1.0684125e-07]]))
    assert np.allclose(basis[:, :, 1, 6], np.array([[8.1380980e-08, 9.0716095e-06, 7.8257020e-08],
                                                    [-8.9917905e-06, 0.0000000e+00, -8.9917905e-06],
                                                    [7.8257020e-08, 9.0716095e-06, 8.1380980e-08]]))
    assert np.allclose(basis[1, 1, :, 7], [-1.49325e-09, 0.00000e+00, -1.49325e-09])


class TestGetFlipMatrix:
    def test_flip_cs(self):
        out = get_flip_matrix('RAS', orders=[1, ])
        assert np.all(out == [1, 1, 1])

    def test_flip_cs_lpi(self):
        out = get_flip_matrix('LPI', orders=[1, ])
        assert np.all(out == [-1, -1, -1])

    def test_flip_cs_order2(self):
        out = get_flip_matrix('LAI', orders=[1, 2])
        assert np.all(out == [1, -1, -1, -1, -1, 1, 1, 1])

    def test_flip_cs_len4(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAIS')

    def test_flip_cs_lap(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAP')

    def test_flip_siemens(self):
        out = get_flip_matrix('LAI', orders=[1, 2, 3], manufacturer='Siemens')
        # TODO: Verify 3rd order
        assert np.all(out == [-1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1])

    def test_flip_ge(self):
        out = get_flip_matrix('LPI', orders=[1, 2], manufacturer='GE')
        assert np.all(out == [-1, -1, -1, 1, 1, 1, 1, 1])

    def test_flip_philips(self):
        out = get_flip_matrix('RPI', orders=[1, 2], manufacturer='PHILIPS')
        assert np.all(out == [1, -1, -1, 1, -1, 1, 1, -1])


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_philips_basis(x, y, z):
    # Off center z axis
    basis = philips_basis(x, y, z)
    # Test for shape
    assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))
    # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    # Philips X axis is AP while Y axis is RL, therefore we check axis 1 for channel 0 and axis 0 for channel 1
    assert np.allclose(basis[1, :, 1, 0], [-42.57748, 0, 42.57748])
    assert np.allclose(basis[:, 1, 1, 1], [42.57748, 0, -42.57748])
    assert np.allclose(basis[1, 1, :, 2], [42.57748, 0, -42.57748])
    assert np.allclose(basis[1, 1, :, 3], [4.25774785e-02, 0, 4.25774785e-02])
    assert np.allclose(basis[1, :, :, 4], np.array([[-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2],
                                                    [0, 0, 0],
                                                    [8.5154957e-02 / 2, 0, -8.5154957e-02 / 2]]))
    assert np.allclose(basis[:, 1, :, 5], np.array([[8.5154957e-02 / 2, 0, -8.5154957e-02 / 2],
                                                    [0, 0, 0],
                                                    [-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2]]))
    assert np.allclose(basis[:, :, 1, 6], np.array([[0, -4.25774785e-02, 0],
                                                    [4.25774785e-02, 0, 4.25774785e-02],
                                                    [0, -4.25774785e-02, 0]]))
    assert np.allclose(basis[:, :, 1, 7], np.array([[-8.51549570e-02, 0, 8.51549570e-02],
                                                    [0, 0, 0],
                                                    [8.51549570e-02, 0, -8.51549570e-02]]))
