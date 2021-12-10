#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import math
import os
import nibabel as nib
import json

import pytest

from shimmingtoolbox.coils.coordinates import generate_meshgrid, phys_gradient, phys_to_vox_gradient, resample_from_to
from shimmingtoolbox.coils.coordinates import get_main_orientation
from shimmingtoolbox import __dir_testing__


fname_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')

nii_fieldmap = nib.load(fname_fieldmap)
nii_anat = nib.load(fname_anat)


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

    affine = nii_fieldmap.affine
    fmap = nii_fieldmap.get_fdata()

    g_x, g_y, g_z = phys_gradient(fmap[..., 0], affine)

    # Test against scaled, non rotated sagittal fieldmap, this should get the same results as phys_gradient
    x_coord, y_coord, z_coord = generate_meshgrid(fmap[..., 0].shape, affine)
    gx_truth = np.zeros_like(fmap[..., 0])
    gy_truth = np.gradient(fmap[..., 0], y_coord[:, 0, 0], axis=0)
    gz_truth = np.gradient(fmap[..., 0], z_coord[0, :, 0], axis=1)

    assert np.all(np.isclose(g_x, gx_truth)) and np.all(np.isclose(g_y, gy_truth)) and np.all(np.isclose(g_z, gz_truth))


def test_phys_to_vox_gradient_synt():
    """Define a previously calculated matrix"""
    img_array = np.expand_dims(np.array([[6 * math.sqrt(2), 4 * math.sqrt(2), 2 * math.sqrt(2)],
                                         [2 * math.sqrt(2), 0, -2 * math.sqrt(2)],
                                         [-2 * math.sqrt(2), -4 * math.sqrt(2), -6 * math.sqrt(2)]]), -1)
    # Define a scaling matrix
    x_vox_spacing = 1
    y_vox_spacing = -2
    scale = np.array([[x_vox_spacing, 0, 0],
                      [0, y_vox_spacing, 0],
                      [0, 0, 1]])

    # Define a rotation matrix
    deg_angle = -10
    rot = np.array([[math.cos(deg_angle * math.pi / 180), -math.sin(deg_angle * math.pi / 180), 0],
                    [math.sin(deg_angle * math.pi / 180), math.cos(deg_angle * math.pi / 180), 0],
                    [0, 0, 1]])

    # Calculate affine matrix
    m_affine = np.dot(rot, scale)
    static_affine = [0, 0, 0, 1]
    affine = np.zeros([4, 4])
    affine[:3, :3] = m_affine
    affine[3, :] = static_affine

    gx_phys, gy_phys, gz_phys = phys_gradient(img_array, affine)  # gx = -5.32, gy = 2.37, gz = 0

    gx_vox, gy_vox, gz_vox = phys_to_vox_gradient(gx_phys, gy_phys, gz_phys, affine)  # gx_vox = -5.66, gy_vox = -1.41

    # Calculate ground truth with the original matrix
    gx_truth = np.gradient(img_array, abs(x_vox_spacing), axis=0)
    gy_truth = np.gradient(img_array, abs(y_vox_spacing), axis=1)
    gz_truth = np.zeros_like(img_array)

    assert np.all(np.isclose(gx_vox, gx_truth)) and \
           np.all(np.isclose(gy_vox, gy_truth)) and \
           np.all(np.isclose(gz_vox, gz_truth))


def test_phys_to_vox_gradient_reel():
    """
    Test the function on real data at 0 degrees of rotation so a ground truth can be calculated with a simple
    gradient calculation since they are parallel. The reel data adds a degree of complexity since it is a sagittal image
    """

    affine = nii_fieldmap.affine
    fmap = nii_fieldmap.get_fdata()

    gx_phys, gy_phys, gz_phys = phys_gradient(fmap[..., 0], affine)

    gx_vox, gy_vox, gz_vox = phys_to_vox_gradient(gx_phys, gy_phys, gz_phys, affine)

    # Test against scaled, non rotated sagittal fieldmap, this should get the same results as phys_gradient
    x_coord, y_coord, z_coord = generate_meshgrid(fmap[..., 0].shape, affine)
    gx_truth = np.zeros_like(fmap[..., 0])
    gy_truth = np.gradient(fmap[..., 0], abs(y_coord[1, 0, 0] - y_coord[0, 0, 0]), axis=0)
    gz_truth = np.gradient(fmap[..., 0], abs(z_coord[0, 1, 0] - z_coord[0, 0, 0]), axis=1)

    assert np.all(np.isclose(gx_truth, gz_vox)) and \
           np.all(np.isclose(gy_truth, gx_vox)) and \
           np.all(np.isclose(gz_truth, gy_vox))


def test_resample_from_to_2d():
    """Test resample_from_to with 2d input."""
    nii_fieldmap_2d = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0, 0], nii_fieldmap.affine)

    nii_resampled = resample_from_to(nii_fieldmap_2d, nii_anat, mode='nearest')

    assert nii_resampled.shape == nii_anat.shape


def test_resample_from_to_3d():
    """Test resample_from_to with 3d input."""
    nii_fieldmap_3d = nib.Nifti1Image(nii_fieldmap.get_fdata()[..., 0], nii_fieldmap.affine)

    nii_resampled = resample_from_to(nii_fieldmap_3d, nii_anat, mode='nearest')

    assert nii_resampled.shape == nii_anat.shape


def test_resample_from_to_4d():
    """Test resample_from_to with 4d input."""
    nii_resampled = resample_from_to(nii_fieldmap, nii_anat, mode='nearest')

    assert nii_resampled.shape == (nii_anat.shape + (nii_fieldmap.shape[3],))


def test_resample_from_to_5d():
    """Test resample_from_to with 5d input. Should return an error."""
    nii_fieldmap_5d = nib.Nifti1Image(np.expand_dims(nii_fieldmap.get_fdata(), -1), nii_fieldmap.affine)

    # This should return an error
    try:
        resample_from_to(nii_fieldmap_5d, nii_anat, mode='nearest')
    except NotImplementedError:
        # If an exception occurs, this is the desired behaviour
        return 0

    # If there isn't an error, then there is a problem
    print('\nWrong dimensions but does not throw an error.')
    assert False


def test_get_main_orientation_tra():
    fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.json')
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    orientation = get_main_orientation(json_data['ImageOrientationPatientDICOM'])

    assert orientation == 'TRA'


def test_get_main_orientation_sag():
    fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.json')
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    orientation = get_main_orientation(json_data['ImageOrientationPatientDICOM'])

    assert orientation == 'SAG'


def test_get_main_orientation_cor():
    cor_orientation = [1, 0, 0, 0, 0, -1]
    orientation = get_main_orientation(cor_orientation)

    assert orientation == 'COR'


def test_get_main_orientation_not_imp():
    cor_orientation = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, -1]

    with pytest.raises(NotImplementedError, match="Ambiguous slice orientation"):
        get_main_orientation(cor_orientation)
