#!/usr/bin/python3
# -*- coding: utf-8 -*
# Deals with coordinate systems, going from voxel-based to physical-based coordinates.

import numpy as np
from nibabel.affines import apply_affine
import math
import nibabel as nib
from nibabel.processing import resample_from_to as nib_resample_from_to


def generate_meshgrid(dim, affine):
    """
    Generate meshgrid of size dim, with coordinate system defined by affine.
    Args:
        dim (tuple): x, y and z dimensions.
        affine (numpy.ndarray): 4x4 affine matrix

    Returns:
        list: List of numpy.ndarray containing meshgrid of coordinates
    """

    nx, ny, nz = dim
    coord_vox = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    coord_phys = [np.zeros_like(coord_vox[0]).astype(float),
                  np.zeros_like(coord_vox[1]).astype(float),
                  np.zeros_like(coord_vox[2]).astype(float)]

    # TODO: Better code
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                coord_phys_list = apply_affine(affine, [coord_vox[i][ix, iy, iz] for i in range(3)])
                for i in range(3):
                    coord_phys[i][ix, iy, iz] = coord_phys_list[i]
    return coord_phys


def phys_gradient(data, affine):
    """Calculate the gradient of ``data`` along physical coordinates defined by ``affine``

    Args:
        data (numpy.ndarray): 3d array containing data to apply gradient
        affine (numpy.ndarray): 4x4 array containing affine transformation

    Returns
        numpy.ndarray: 3D matrix containing the gradient along the x direction in the physical coordinate system
        numpy.ndarray: 3D matrix containing the gradient along the y direction in the physical coordinate system
        numpy.ndarray: 3D matrix containing the gradient along the z direction in the physical coordinate system
    """

    x_vox = 0
    y_vox = 1
    z_vox = 2

    # Calculate the spacing along the different voxel axis
    x_vox_spacing = math.sqrt((affine[0, x_vox] ** 2) + (affine[1, x_vox] ** 2) + (affine[2, x_vox] ** 2))
    y_vox_spacing = math.sqrt((affine[0, y_vox] ** 2) + (affine[1, y_vox] ** 2) + (affine[2, y_vox] ** 2))
    z_vox_spacing = math.sqrt((affine[0, z_vox] ** 2) + (affine[1, z_vox] ** 2) + (affine[2, z_vox] ** 2))

    # Compute the gradient along the different voxel axis
    if data.shape[x_vox] != 1:
        x_vox_gradient = np.gradient(data, x_vox_spacing, axis=x_vox)
    else:
        x_vox_gradient = np.zeros_like(data)

    if data.shape[y_vox] != 1:
        y_vox_gradient = np.gradient(data, y_vox_spacing, axis=y_vox)
    else:
        y_vox_gradient = np.zeros_like(data)

    if data.shape[z_vox] != 1:
        z_vox_gradient = np.gradient(data, z_vox_spacing, axis=z_vox)
    else:
        z_vox_gradient = np.zeros_like(data)

    # Compute the gradient along the physical axis
    x_gradient = (x_vox_gradient * affine[0, x_vox] / x_vox_spacing) + \
                 (y_vox_gradient * affine[0, y_vox] / y_vox_spacing) + \
                 (z_vox_gradient * affine[0, z_vox] / z_vox_spacing)
    y_gradient = (x_vox_gradient * affine[1, x_vox] / x_vox_spacing) + \
                 (y_vox_gradient * affine[1, y_vox] / y_vox_spacing) + \
                 (z_vox_gradient * affine[1, z_vox] / z_vox_spacing)
    z_gradient = (x_vox_gradient * affine[2, x_vox] / x_vox_spacing) + \
                 (y_vox_gradient * affine[2, y_vox] / y_vox_spacing) + \
                 (z_vox_gradient * affine[2, z_vox] / z_vox_spacing)

    return x_gradient, y_gradient, z_gradient


def phys_to_vox_gradient(gx, gy, gz, affine):
    """
    Calculate the gradient along the voxel coordinates defined by ``affine`` with gradients in the physical
    coordinate system

    Args:
        gx (numpy.ndarray): 3D matrix containing the gradient along the x direction in the physical coordinate system
        gy (numpy.ndarray): 3D matrix containing the gradient along the y direction in the physical coordinate system
        gz (numpy.ndarray): 3D matrix containing the gradient along the z direction in the physical coordinate system
        affine (numpy.ndarray): 4x4 array containing affine transformation

    Returns:
        numpy.ndarray: 3D matrix containing the gradient along the x direction in the voxel coordinate system
        numpy.ndarray: 3D matrix containing the gradient along the y direction in the voxel coordinate system
        numpy.ndarray: 3D matrix containing the gradient along the z direction in the voxel coordinate system
    """

    x_vox = 0
    y_vox = 1
    z_vox = 2

    # Calculate the spacing along the different voxel axis
    x_vox_spacing = math.sqrt((affine[0, x_vox] ** 2) + (affine[1, x_vox] ** 2) + (affine[2, x_vox] ** 2))
    y_vox_spacing = math.sqrt((affine[0, y_vox] ** 2) + (affine[1, y_vox] ** 2) + (affine[2, y_vox] ** 2))
    z_vox_spacing = math.sqrt((affine[0, z_vox] ** 2) + (affine[1, z_vox] ** 2) + (affine[2, z_vox] ** 2))

    inv_affine = np.linalg.inv(affine[:3, :3])

    gx_vox = (gx * inv_affine[0, x_vox] * x_vox_spacing) + \
             (gy * inv_affine[0, y_vox] * x_vox_spacing) + \
             (gz * inv_affine[0, z_vox] * x_vox_spacing)
    gy_vox = (gx * inv_affine[1, x_vox] * y_vox_spacing) + \
             (gy * inv_affine[1, y_vox] * y_vox_spacing) + \
             (gz * inv_affine[1, z_vox] * y_vox_spacing)
    gz_vox = (gx * inv_affine[2, x_vox] * z_vox_spacing) + \
             (gy * inv_affine[2, y_vox] * z_vox_spacing) + \
             (gz * inv_affine[2, z_vox] * z_vox_spacing)

    return gx_vox, gy_vox, gz_vox


def resample_from_to(nii_from_img, nii_to_vox_map, order=2, mode='nearest', cval=0., out_class=nib.Nifti1Image):
    """ Wrapper to nibabel's ``resample_from_to`` function. Resample image `from_img` to mapped voxel space
    `to_vox_map`. The wrapper adds support for 2D input data (adds a singleton) and for 4D time series.
    For more info, refer to nibabel.processing.resample_from_to.

    Args:
        nii_from_img (nibabel.Nifti1Image): Nibabel object with 2D, 3D or 4D array. The 4d case will be treated as a
                                            timeseries.
        nii_to_vox_map (nibabel.Nifti1Image):
        order (int): Refer to nibabel.processing.resample_from_to
        mode (str): Refer to nibabel.processing.resample_from_to
        cval (scalar): Refer to nibabel.processing.resample_from_to
        out_class: Refer to nibabel.processing.resample_from_to

    Returns:
        nibabel.Nifti1Image: Return a Nibabel object with the resampled data. The 4d case will have an extra dimension
                             for the different time points.

    """

    from_img = nii_from_img.get_fdata()
    if from_img.ndim == 2:
        nii_from_img_3d = nib.Nifti1Image(np.expand_dims(from_img, -1), nii_from_img.affine)
        nii_resampled = nib_resample_from_to(nii_from_img_3d, nii_to_vox_map, order=order, mode=mode, cval=cval,
                                             out_class=out_class)

    elif from_img.ndim == 3:
        nii_resampled = nib_resample_from_to(nii_from_img, nii_to_vox_map, order=order, mode=mode, cval=cval,
                                             out_class=out_class)

    elif from_img.ndim == 4:
        nt = from_img.shape[3]
        resampled_4d = np.zeros(nii_to_vox_map.shape + (nt,))
        for it in range(from_img.shape[3]):
            nii_from_img_3d = nib.Nifti1Image(from_img[..., it], nii_from_img.affine)
            nii_resampled_3d = nib_resample_from_to(nii_from_img_3d, nii_to_vox_map, order=order, mode=mode, cval=cval,
                                                    out_class=out_class)
            resampled_4d[..., it] = nii_resampled_3d.get_fdata()
        nii_resampled = nib.Nifti1Image(resampled_4d, nii_to_vox_map.affine)

    else:
        raise NotImplementedError("Dimensions of input can only be 2D, 3D or 4D")

    return nii_resampled
