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
class TestSiemensBasis:
    def test_siemens_basis(self, x, y, z):
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
        assert np.allclose(basis[1, :, :, 9], np.array([[-2.12887393e-08, -2.17259887e-24, 2.12887393e-08],
                                                        [4.25774785e-08, 0.00000000e+00, -4.25774785e-08],
                                                        [-2.12887393e-08, -2.17259887e-24, 2.12887393e-08]]))
        assert np.allclose(basis[:, 1, :, 10], np.array([[1.20427295e-07, -4.01424317e-08, 1.20427295e-07],
                                                         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
                                                         [-1.20427295e-07, 4.01424317e-08, -1.20427295e-07]]))
        assert np.allclose(basis[1, :, :, 11], np.array([[-1.20427295e-07, 4.01424317e-08, -1.20427295e-07],
                                                         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
                                                         [1.20427295e-07, -4.01424317e-08, 1.20427295e-07]]))
        assert np.allclose(basis[:2, :2, :2, 12], np.array([[[1.47480902e-23, -0.00000000e+00],
                                                            [1.20427295e-07, -0.00000000e+00]],
                                                           [[-1.20427295e-07, 0.00000000e+00],
                                                            [0.00000000e+00, 0.00000000e+00]]]))

    def test_siemens_basis_order1(self, x, y, z):
        basis = siemens_basis(x, y, z, orders=(1,))
        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 3)))
        # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
        assert np.allclose(basis[:, 1, 1, 0], [4.25774785e-02, 0, -4.25774785e-02])
        assert np.allclose(basis[1, :, 1, 1], [-4.25774785e-02, 0, 4.25774785e-02])
        assert np.allclose(basis[1, 1, :, 2], [4.25774785e-02, 0, -4.25774785e-02])

    def test_siemens_basis_order2(self, x, y, z):
        basis = siemens_basis(x, y, z, orders=(2,))
        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 5)))
        # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
        assert np.allclose(basis[1, 1, :, 0], [4.25774785e-05, 0.00000000e+00, 4.25774785e-05])
        assert np.allclose(basis[:, 1, :, 1], np.array([[8.5154957e-05, 0.0000000e+00, -8.5154957e-05],
                                                        [-0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                                                        [-8.5154957e-05, -0.0000000e+00, 8.5154957e-05]]))
        assert np.allclose(basis[1, :, :, 2], np.array([[-8.5154957e-05, -0.0000000e+00, 8.5154957e-05],
                                                        [0.0000000e+00, -0.0000000e+00, -0.0000000e+00],
                                                        [8.5154957e-05, 0.0000000e+00, -8.5154957e-05]]))
        assert np.allclose(basis[:, :, 1, 3], np.array([[0, 4.25774785e-05, 0],
                                                        [-4.25774785e-05, 0.00000000e+00, -4.25774785e-05],
                                                        [0, 4.25774785e-05, 0]]))
        assert np.allclose(basis[:, :, 1, 4], np.array([[-8.51549570e-05, 0, 8.51549570e-05],
                                                        [0, 0, 0],
                                                        [8.51549570e-05, -0.00000000e+00, -8.51549570e-05]]))

    def test_create_siemens_basis_order3(self, x, y, z):
        basis = siemens_basis(x, y, z, orders=(3,))
        assert np.all(basis.shape == (3, 3, 3, 4))
        assert np.allclose(basis[1, :, :, 0], np.array([[-2.12887393e-08, -2.17259887e-24, 2.12887393e-08],
                                                        [4.25774785e-08, 0.00000000e+00, -4.25774785e-08],
                                                        [-2.12887393e-08, -2.17259887e-24, 2.12887393e-08]]))
        assert np.allclose(basis[:, 1, :, 1], np.array([[1.20427295e-07, -4.01424317e-08, 1.20427295e-07],
                                                         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
                                                         [-1.20427295e-07, 4.01424317e-08, -1.20427295e-07]]))
        assert np.allclose(basis[1, :, :, 2], np.array([[-1.20427295e-07, 4.01424317e-08, -1.20427295e-07],
                                                         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
                                                         [1.20427295e-07, -4.01424317e-08, 1.20427295e-07]]))
        assert np.allclose(basis[:2, :2, :2, 3], np.array([[[1.47480902e-23, -0.00000000e+00],
                                                            [1.20427295e-07, -0.00000000e+00]],
                                                           [[-1.20427295e-07, 0.00000000e+00],
                                                            [0.00000000e+00, 0.00000000e+00]]]))

    def test_create_siemens_basis_order4(self, x, y, z):
        with pytest.raises(NotImplementedError, match="Spherical harmonics not implemented for order 4 and up"):
            siemens_basis(x, y, z, orders=(4,))


@pytest.mark.parametrize('x,y,z', dummy_data)
class TestGEBasis:
    def test_ge_basis(self, x, y, z):
        basis = ge_basis(x, y, z, (0, 1, 2))

        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 9)))
        # x, y, z, xy, zy, zx, X2 - Y2, z2
        assert np.allclose(basis[:, 1, 1, 0], [-1, -1, -1])
        assert np.allclose(basis[:, 1, 1, 1], [0.00425775, 0, -0.00425775])
        assert np.allclose(basis[1, :, 1, 2], [0.00425775, 0, -0.00425775])
        assert np.allclose(basis[1, 1, :, 3], [0.00425775, 0, -0.00425775])
        assert np.allclose(basis[:, :, 1, 4], np.array([[-0.30583688, -0.30581601, -0.30581504],
                                                        [-0.30577086, -0.30575, -0.30574904],
                                                        [-0.30572506, -0.3057042, -0.30570324]]))
        assert np.allclose(basis[1, :, :, 5], np.array([[-5.6703165e-05, -5.0049942e-05, -4.3501609e-05],
                                                        [-5.9169445e-05, -5.2430000e-05, -4.5795445e-05],
                                                        [-6.1538609e-05, -5.4712942e-05, -4.7992165e-05]]))
        assert np.allclose(basis[:, 1, :, 6], np.array([[0.00120057, 0.00119278, 0.00118491],
                                                        [0.00120356, 0.0011958, 0.00118797],
                                                        [0.00120663, 0.0011989, 0.0011911]]))
        assert np.allclose(basis[:, :, 1, 7], np.array([[0.00045976, 0.00048539, 0.00049299],
                                                        [0.00046216, 0.00048776, 0.00049534],
                                                        [0.00048266, 0.00050823, 0.00051578]]))
        assert np.allclose(basis[1, 1, :, 8], [0.00040881, 0.00042614, 0.00044346])

    def test_ge_basis_order1(self, x, y, z):
        basis = ge_basis(x, y, z, orders=(1,))

        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 3)))
        # x, y, z
        assert np.allclose(basis[:, 1, 1, 0], [0.00425775, 0, -0.00425775])
        assert np.allclose(basis[1, :, 1, 1], [0.00425775, 0, -0.00425775])
        assert np.allclose(basis[1, 1, :, 2], [0.00425775, 0, -0.00425775])

    def test_ge_basis_order2(self, x, y, z):
        basis = ge_basis(x, y, z, orders=(2,))

        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 5)))
        # xy, zy, zx, X2 - Y2, z2
        assert np.allclose(basis[:, :, 1, 0], np.array([[-0.30583688, -0.30581601, -0.30581504],
                                                        [-0.30577086, -0.30575, -0.30574904],
                                                        [-0.30572506, -0.3057042, -0.30570324]]))
        assert np.allclose(basis[1, :, :, 1], np.array([[-5.6703165e-05, -5.0049942e-05, -4.3501609e-05],
                                                        [-5.9169445e-05, -5.2430000e-05, -4.5795445e-05],
                                                        [-6.1538609e-05, -5.4712942e-05, -4.7992165e-05]]))
        assert np.allclose(basis[:, 1, :, 2], np.array([[0.00120057, 0.00119278, 0.00118491],
                                                        [0.00120356, 0.0011958, 0.00118797],
                                                        [0.00120663, 0.0011989, 0.0011911]]))
        assert np.allclose(basis[:, :, 1, 3], np.array([[0.00045976, 0.00048539, 0.00049299],
                                                        [0.00046216, 0.00048776, 0.00049534],
                                                        [0.00048266, 0.00050823, 0.00051578]]))
        assert np.allclose(basis[1, 1, :, 4], [0.00040881, 0.00042614, 0.00044346])


@pytest.mark.parametrize('x,y,z', dummy_data)
class TestPhilipsBasis:
    def test_philips_basis(self, x, y, z):
        # Off center z axis
        basis = philips_basis(x, y, z, orders=(0, 1, 2))
        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 9)))
        # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
        # Philips X axis is AP while Y axis is RL, therefore we check axis 1 for channel 0 and axis 0 for channel 1
        assert np.allclose(basis[:, 1, 1, 0], [-1, -1, -1])
        assert np.allclose(basis[1, :, 1, 1], [-42.57748, 0, 42.57748])
        assert np.allclose(basis[:, 1, 1, 2], [42.57748, 0, -42.57748])
        assert np.allclose(basis[1, 1, :, 3], [42.57748, 0, -42.57748])
        assert np.allclose(basis[1, 1, :, 4], [4.25774785e-02, 0, 4.25774785e-02])
        assert np.allclose(basis[1, :, :, 5], np.array([[-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2],
                                                        [0, 0, 0],
                                                        [8.5154957e-02 / 2, 0, -8.5154957e-02 / 2]]))
        assert np.allclose(basis[:, 1, :, 6], np.array([[8.5154957e-02 / 2, 0, -8.5154957e-02 / 2],
                                                        [0, 0, 0],
                                                        [-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2]]))
        assert np.allclose(basis[:, :, 1, 7], np.array([[0, -4.25774785e-02, 0],
                                                        [4.25774785e-02, 0, 4.25774785e-02],
                                                        [0, -4.25774785e-02, 0]]))
        assert np.allclose(basis[:, :, 1, 8], np.array([[-8.51549570e-02, 0, 8.51549570e-02],
                                                        [0, 0, 0],
                                                        [8.51549570e-02, 0, -8.51549570e-02]]))

    def test_philips_basis_order1(self, x, y, z):
        # Off center z axis
        basis = philips_basis(x, y, z, orders=(1,))
        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 3)))
        # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
        # Philips X axis is AP while Y axis is RL, therefore we check axis 1 for channel 0 and axis 0 for channel 1
        assert np.allclose(basis[1, :, 1, 0], [-42.57748, 0, 42.57748])
        assert np.allclose(basis[:, 1, 1, 1], [42.57748, 0, -42.57748])
        assert np.allclose(basis[1, 1, :, 2], [42.57748, 0, -42.57748])

    def test_philips_basis_order2(self, x, y, z):
        # Off center z axis
        basis = philips_basis(x, y, z, orders=(2,))
        # Test for shape
        assert (np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 5)))
        # X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
        # Philips X axis is AP while Y axis is RL, therefore we check axis 1 for channel 0 and axis 0 for channel 1
        assert np.allclose(basis[1, 1, :, 0], [4.25774785e-02, 0, 4.25774785e-02])
        assert np.allclose(basis[1, :, :, 1], np.array([[-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2],
                                                        [0, 0, 0],
                                                        [8.5154957e-02 / 2, 0, -8.5154957e-02 / 2]]))
        assert np.allclose(basis[:, 1, :, 2], np.array([[8.5154957e-02 / 2, 0, -8.5154957e-02 / 2],
                                                        [0, 0, 0],
                                                        [-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2]]))
        assert np.allclose(basis[:, :, 1, 3], np.array([[0, -4.25774785e-02, 0],
                                                        [4.25774785e-02, 0, 4.25774785e-02],
                                                        [0, -4.25774785e-02, 0]]))
        assert np.allclose(basis[:, :, 1, 4], np.array([[-8.51549570e-02, 0, 8.51549570e-02],
                                                        [0, 0, 0],
                                                        [8.51549570e-02, 0, -8.51549570e-02]]))


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
