#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image mask with shape API
"""

import numpy as np


def shape_square(data, len_dim1, len_dim2, center_dim1=None, center_dim2=None):
    """
    Creates a square mask. Returns mask with the same shape as `data`.

    Args:
        data (numpy.ndarray): Data to mask, must be 2 dimensional array.
        len_dim1 (int): Length of the side of the square along first dimension (in pixels).
        len_dim2 (int): Length of the side of the square along second dimension (in pixels).
        center_dim1 (int): Center of the square along first dimension (in pixels). If no center is provided, the middle
                           is used.
        center_dim2 (int): Center of the square along second dimension (in pixels). If no center is provided, the middle
                           is used.
    Returns:
        numpy.ndarray: Mask with booleans. True where the square is located and False in the background.
    """

    # Only takes data with 2 dimensions
    if data.ndim != 2:
        raise ValueError("shape_square only allows for 2 dimensions")

    # Default to middle
    if center_dim1 is None:
        center_dim1 = int(data.shape[0] / 2)
    if center_dim2 is None:
        center_dim2 = int(data.shape[1] / 2)

    # Create a meshgrid of the size of input data
    dim1_v, dim2_v = np.meshgrid(np.arange(0, data.shape[0]), np.arange(0, data.shape[1]), indexing='ij')

    # Create the rectangle by allowing values from greater or lower than specified inputs
    dim1_v_logical = np.logical_and(dim1_v >= center_dim1 - int(np.floor(len_dim1 / 2)),
                                    dim1_v < center_dim1 + int(np.ceil(len_dim1 / 2)))
    dym2_v_logical = np.logical_and(dim2_v >= center_dim2 - int(np.floor(len_dim2 / 2)),
                                    dim2_v < center_dim2 + int(np.ceil(len_dim2 / 2)))
    mask = dim1_v_logical & dym2_v_logical

    return mask


def shape_cube(data, len_dim1, len_dim2, len_dim3, center_dim1=None, center_dim2=None, center_dim3=None):
    """
    Creates a cube mask. Returns mask with the same shape as `data`.

    Args:
        data (numpy.ndarray): Data to mask, must be 3 dimensional array.
        len_dim1 (int): Length of the side of the square along first dimension (in pixels).
        len_dim2 (int): Length of the side of the square along second dimension (in pixels).
        len_dim3 (int): Length of the side of the square along third dimension (in pixels).
        center_dim1 (int): Center of the square along first dimension (in pixels). If no center is provided, the middle
                           is used.
        center_dim2 (int): Center of the square along second dimension (in pixels). If no center is provided, the middle
                           is used.
        center_dim3 (int): Center of the square along third dimension (in pixels). If no center is provided, the middle
                           is used.
    Returns:
        numpy.ndarray: Mask with booleans. True where the cube is located and False in the background.
    """

    # Only takes data with 3 dimensions
    if data.ndim != 3:
        raise ValueError("shape_cube only allows for 2 dimensions")

    # Default to middle
    if center_dim1 is None:
        center_dim1 = int(data.shape[0] / 2)
    if center_dim2 is None:
        center_dim2 = int(data.shape[1] / 2)
    if center_dim3 is None:
        center_dim3 = int(data.shape[2] / 2)

    # Create a meshgrid of the size of input data
    dim1_v, dim2_v, dim3_v = np.meshgrid(np.arange(0, data.shape[0]),
                                         np.arange(0, data.shape[1]),
                                         np.arange(0, data.shape[2]), indexing='ij')

    # Create the rectangle by allowing values from greater or lower than specified inputs
    dim1_v_logical = np.logical_and(dim1_v >= center_dim1 - int(np.floor(len_dim1 / 2)),
                                    dim1_v < center_dim1 + int(np.ceil(len_dim1 / 2)))
    dym2_v_logical = np.logical_and(dim2_v >= center_dim2 - int(np.floor(len_dim2 / 2)),
                                    dim2_v < center_dim2 + int(np.ceil(len_dim2 / 2)))
    dim3_v_logical = np.logical_and(dim3_v >= center_dim3 - int(np.floor(len_dim3 / 2)),
                                    dim3_v < center_dim3 + int(np.ceil(len_dim3 / 2)))
    mask = dim1_v_logical & dym2_v_logical & dim3_v_logical

    return mask


def shape_sphere(data, radius, center_dim1=None, center_dim2=None, center_dim3=None):
    """
    Creates a spherical mask. Returns mask with the same shape as `data`.

    Args:
        data (numpy.ndarray): Data to mask, must be 3 dimensional array.
        radius (int): Radius of the sphere (in pixels).
        center_dim1 (int): Center of the sphere along the first dimension (in pixels). If no center is provided, the
                           middle is used.
        center_dim2 (int): Center of the sphere along the second dimension (in pixels). If no center is provided, the
                           middle is used.
        center_dim3 (int): Center of the sphere along the third dimension (in pixels). If no center is provided, the
                           middle is used.
    Returns:
        numpy.ndarray: Mask with booleans. True where the sphere is located and False in the background.
    """

    if data.ndim != 3:
        raise ValueError("shape_sphere only allows for 3 dimensions")

    shape = data.shape

    # Defaults to the center of the array if no center is provided
    if center_dim1 is None:
        center_dim1 = shape[0] // 2
    if center_dim2 is None:
        center_dim2 = shape[1] // 2
    if center_dim3 is None:
        center_dim3 = shape[2] // 2

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center_dim1) ** 2 + (y - center_dim2) ** 2 + (z - center_dim3) ** 2)
    mask = dist <= radius

    return mask


shape_mask = {'square': shape_square, 'cube': shape_cube, 'sphere': shape_sphere}


def shapes(data, shape, **kargs):
    """
    Wrapper to different shape masking functions.

    Args:
        data (numpy.ndarray): Data to mask.
        shape (str): Shape to mask, implemented shapes include: {'square', 'cube', 'sphere'}.
        **kargs: Refer to the specific function in this file for the specific arguments for each shape.
                 See example section for more details.

    Returns:
        numpy.ndarray: Mask with booleans. True where the shape is located and False in the background.

    Examples:
        >>> dummy_data = np.ones([4,3])
        >>> dummy_mask = shapes(dummy_data, 'square', center_dim1=1, center_dim2=1, len_dim1=1, len_dim2=3)
    """

    mask_info = {}
    for key, value in kargs.items():
        mask_info[key] = value

    mask = shape_mask[shape](data, **mask_info)

    return mask
