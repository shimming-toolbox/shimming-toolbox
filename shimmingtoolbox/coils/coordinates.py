#!/usr/bin/python3
# -*- coding: utf-8 -*
# Deals with coordinate systems, going from voxel-based to physical-based coordinates.

# TODO: create a test for this API

import numpy as np
from nibabel.affines import apply_affine


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
