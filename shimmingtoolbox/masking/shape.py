#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image mask with shape API
# TODO: Make sure specified center and length fit within data.shape
"""

import numpy as np


def shape_square(data, center_x, center_y, len_x, len_y):
    """
    Create a square mask

    Args:

    Returns:
        nd.array: mask
    """

    # Only takes data as an X,Y array
    if data.ndim != 2:
        raise RuntimeError('shape_square only allows for 2 dimensions')

    if center_x is None:
        center_x = int(data.shape[0] / 2)
    if center_y is None:
        center_y = int(data.shape[1] / 2)

    # Create a meshgrid of the size of input data
    yv, xv = np.meshgrid(np.arange(0, data.shape[1]), np.arange(0, data.shape[0]))

    # Create the rectangle by allowing values from greater or lower than specified inputs
    xv_logical = np.logical_and(xv >= center_x - int(np.floor(len_x / 2)), xv < center_x + int(np.ceil(len_x / 2)))
    yv_logical = np.logical_and(yv >= center_y - int(np.floor(len_y / 2)), yv < center_y + int(np.ceil(len_y / 2)))
    mask = np.logical_and(xv_logical, yv_logical)

    return mask


def shape_cube(data, center_x, center_y, center_z, len_x, len_y, len_z):
    """
    Create a cube mask

    Args:

    Returns:
        nd.array: mask
    """

    # Only takes data as an X,Y array
    if data.ndim != 3:
        raise RuntimeError('shape_square only allows for 2 dimensions')

    if center_x is None:
        center_x = int(data.shape[0] / 2)
    if center_y is None:
        center_y = int(data.shape[1] / 2)
    if center_z is None:
        center_z = int(data.shape[2] / 2)

    # Create a meshgrid of the size of input data
    yv, xv, zv = np.meshgrid(np.arange(0, data.shape[1]), np.arange(0, data.shape[0]), np.arange(0, data.shape[2]))

    # Create the rectangle by allowing values from greater or lower than specified inputs
    xv_logical = np.logical_and(xv >= center_x - int(np.floor(len_x / 2)), xv < center_x + int(np.ceil(len_x / 2)))
    yv_logical = np.logical_and(yv >= center_y - int(np.floor(len_y / 2)), yv < center_y + int(np.ceil(len_y / 2)))
    zv_logical = np.logical_and(zv >= center_z - int(np.floor(len_z / 2)), zv < center_z + int(np.ceil(len_z / 2)))
    mask = np.logical_and(xv_logical, yv_logical, zv_logical)

    return mask


def shape_disk(data, center_x, center_y, radius):
    pass


def shape_sphere(data, center_x, center_y, center_z, radius):
    pass


shape_mask = {'square': shape_square, 'cube': shape_cube, 'disk': shape_disk, 'sphere': shape_sphere}


def shape(data, shape, **kargs):
    # center_x=None, center_y=None, center_z=None, len_x=None, len_y=None, len_z=None, radius=None
    """
    Wrapper to different shape masking algos

    Args:

    Returns:
        nd.array: mask
    """

    mask_info = {}
    for key, value in kargs.items():
        mask_info[key] = value

    mask = shape_mask[shape](data, **mask_info)

    return mask
