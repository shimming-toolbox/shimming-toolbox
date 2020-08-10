#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image mask with shape API
"""

import numpy as np


def shape(data, shape, center_x=None, center_y=None, center_z=None, len_x=None, len_y=None, len_z=None, radius=None):
    """
    Wrapper to different shape masking algos

    Args:

    Returns:
        nd.array: mask
    """

    # allowed_shapes = {'square', 'cube', 'disk', 'ball'}

    # Make sure input params are valid
    if data.ndim not in [2, 3]:
        raise RuntimeError('Number of dimensions not supported')

    # TODO: Make sure specified center and length fit within data.shape

    # Create mask with all zeros
    mask = np.zeros_like(data)

    if shape == 'square':
        mask = shape_square(data, center_x, center_y, len_x, len_y)
    elif shape == 'cube':
        # TODO
        pass
    elif shape == 'disk':
        # TODO
        pass
    elif shape == 'ball':
        # TODO
        pass
    else:
        raise RuntimeError('Input shapes not part of the allowed shapes for masking')

    return mask


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

    mask = np.zeros_like(data)
    mask[center_y - int(np.floor(len_y / 2)):center_y + int(np.ceil(len_y / 2)), center_x - int(np.floor(len_x / 2)):center_x + int(np.ceil(len_x / 2))] = 1

    return mask
