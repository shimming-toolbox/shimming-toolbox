#!/usr/bin/python3
# -*- coding: utf-8 -*
# Deals with coordinate systems, going from voxel-based to physical-based coordinates.

import numpy as np
from nibabel.affines import apply_affine
import math


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
    """

    x_vox = 0
    y_vox = 1
    z_vox = 2

    x_vox_spacing = math.sqrt((affine[x_vox, 0] ** 2) + (affine[x_vox, 1] ** 2) + (affine[x_vox, 2] ** 2))
    y_vox_spacing = math.sqrt((affine[y_vox, 0] ** 2) + (affine[y_vox, 1] ** 2) + (affine[y_vox, 2] ** 2))
    z_vox_spacing = math.sqrt((affine[z_vox, 0] ** 2) + (affine[z_vox, 1] ** 2) + (affine[z_vox, 2] ** 2))

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

    x_gradient = ((x_vox_gradient * (affine[x_vox, 0] / x_vox_spacing)) +
                  (y_vox_gradient * (affine[x_vox, 1] / y_vox_spacing)) +
                  (z_vox_gradient * (affine[x_vox, 2] / z_vox_spacing)))
    y_gradient = ((x_vox_gradient * (affine[y_vox, 0] / x_vox_spacing)) +
                  (y_vox_gradient * (affine[y_vox, 1] / y_vox_spacing)) +
                  (z_vox_gradient * (affine[y_vox, 2] / z_vox_spacing)))
    z_gradient = ((x_vox_gradient * (affine[z_vox, 2] / x_vox_spacing)) +
                  (y_vox_gradient * (affine[z_vox, 1] / y_vox_spacing)) +
                  (z_vox_gradient * (affine[z_vox, 2] / z_vox_spacing)))

    return x_gradient, y_gradient, z_gradient
