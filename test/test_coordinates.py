#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import math
import os
import nibabel as nib

from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.coils.coordinates import phys_gradient
from shimmingtoolbox import __dir_testing__


def test_generate_meshgrid():
    """Test to verify generate_meshgrid outputs the correct scanner coordinates from input voxels"""

    affine = np.array([[0., 0.,    3., -3.61445999],
                      [-2.91667008, 0., 0., 101.76699829],
                      [0., 2.91667008, 0., -129.85464478],
                      [0., 0., 0., 1.]])

    nx = 2
    ny = 2
    nz = 2
    coord = generate_meshgrid((nx, ny, nz), affine)

    expected = [np.array([[[-3.61445999, -0.61445999],
                           [-3.61445999, -0.61445999]],
                          [[-3.61445999, -0.61445999],
                           [-3.61445999, -0.61445999]]]),
                np.array([[[101.76699829, 101.76699829],
                           [101.76699829, 101.76699829]],
                          [[98.85032821,  98.85032821],
                           [98.85032821,  98.85032821]]]),
                np.array([[[-129.85464478, -129.85464478],
                           [-126.9379747, -126.9379747]],
                          [[-129.85464478, -129.85464478],
                           [-126.9379747, -126.9379747]]])]

    assert(np.all(np.isclose(coord, expected)))


def test_phys_gradient_synt():
    """Define a previously calculated matrix (matrix was defined at 45 deg for gx=-6, gy=2)"""
    img_array = np.expand_dims(np.array([[6 * math.sqrt(2), 4 * math.sqrt(2), 2 * math.sqrt(2)],
                                         [2 * math.sqrt(2), 0, -2 * math.sqrt(2)],
                                         [-2 * math.sqrt(2), -4 * math.sqrt(2), -6 * math.sqrt(2)]]), -1)
    # Define a scaling matrix
    scale = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    # Define a rotation matrix
    deg_angle = -45
    rot = np.array([[math.cos(deg_angle * math.pi / 180), -math.sin(deg_angle * math.pi / 180), 0],
                    [math.sin(deg_angle * math.pi / 180), math.cos(deg_angle * math.pi / 180), 0],
                    [0, 0, 1]])

    # Calculate affine matrix
    m_affine = np.dot(rot, scale)
    static_affine = [0, 0, 0, 1]
    affine = np.zeros([4, 4])
    affine[:3, :3] = m_affine
    affine[3, :] = static_affine

    g_x, g_y, g_z = phys_gradient(img_array, affine)  # gx = -6, gy = 2, gz = 0

    assert np.all(np.isclose(g_x, -6)) and np.all(np.isclose(g_y, 2)) and np.all(np.isclose(g_z, 0))


def test_phys_gradient_synt_scaled():
    """Define a previously calculated matrix (matrix was defined at 45 deg for gx=-3, gy=1)"""
    img_array = np.expand_dims(np.array([[6 * math.sqrt(2), 4 * math.sqrt(2), 2 * math.sqrt(2)],
                                         [2 * math.sqrt(2), 0, -2 * math.sqrt(2)],
                                         [-2 * math.sqrt(2), -4 * math.sqrt(2), -6 * math.sqrt(2)]]), -1)
    # Define a scaling matrix
    scale = np.array([[2, 0, 0],
                      [0, 2, 0],
                      [0, 0, 2]])

    # Define a rotation matrix
    deg_angle = -45
    rot = np.array([[math.cos(deg_angle * math.pi / 180), -math.sin(deg_angle * math.pi / 180), 0],
                    [math.sin(deg_angle * math.pi / 180), math.cos(deg_angle * math.pi / 180), 0],
                    [0, 0, 1]])

    # Calculate affine matrix
    m_affine = np.dot(rot, scale)
    static_affine = [0, 0, 0, 1]
    affine = np.zeros([4, 4])
    affine[:3, :3] = m_affine
    affine[3, :] = static_affine

    g_x, g_y, g_z = phys_gradient(img_array, affine)  # gx = -6, gy = 2, gz = 0

    assert np.all(np.isclose(g_x, -3)) and np.all(np.isclose(g_y, 1)) and np.all(np.isclose(g_z, 0))


def test_phys_gradient_reel():
    """
    Test the function on real data at 0 degrees of rotation so a ground truth can be calculated with a simple
    gradient calculation since they are parallel. The reel data adds a degree of complexity since it is a sagittal image
    """

    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')

    nii_fieldmap = nib.load(fname_fieldmap)

    affine = nii_fieldmap.affine
    fmap = nii_fieldmap.get_fdata()

    g_x, g_y, g_z = phys_gradient(fmap[..., 0], affine)

    # Test against scaled, non rotated sagittal fieldmap, this should get the same results as phys_gradient
    x_coord, y_coord, z_coord = generate_meshgrid(fmap[..., 0].shape, affine)
    gx_truth = np.zeros_like(fmap[..., 0])
    gy_truth = np.gradient(fmap[..., 0], y_coord[:, 0, 0], axis=0)
    gz_truth = np.gradient(fmap[..., 0], z_coord[0, :, 0], axis=1)

    assert np.all(np.isclose(g_x, gx_truth)) and np.all(np.isclose(g_y, gy_truth)) and np.all(np.isclose(g_z, gz_truth))
