#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.morphology import binary_dilation, disk

from shimmingtoolbox.masking.shapes import shape_square

def test_basic_soft_square_mask(data, soft_width, soft_value, len_dim1, len_dim2, center_dim1=None, center_dim2=None) :
    """
    Creates a basic square softmask. Returns softmask with the same shape as `data`.

    Args:
        data (numpy.ndarray): Data to mask, must be 2 dimensional array.
        soft_width (int): Width of the soft zone (in pixels).
        soft_value (float): Value of the intexnsity of the pixels in the soft zone.
        len_dim1 (int): Length of the side of the square along first dimension (in pixels).
        len_dim2 (int): Length of the side of the square along second dimension (in pixels).
        center_dim1 (int): Center of the square along first dimension (in pixels). If no center is provided, the middle
                           is used.
        center_dim2 (int): Center of the square along second dimension (in pixels). If no center is provided, the middle
                           is used.
    Returns:
        numpy.ndarray: Mask with floats.
    """
    bin_square_mask = shape_square(data, len_dim1, len_dim2, center_dim1, center_dim2)
    dilated_mask = binary_dilation(bin_square_mask, disk(soft_width)).astype(float)
    difference = dilated_mask - bin_square_mask
    soft_square_mask = bin_square_mask + difference * soft_value
    return soft_square_mask

def test_soft_square_mask(data, soft_width, len_dim1, len_dim2, center_dim1=None, center_dim2=None) :
    """
    Creates a square softmask. Returns softmask with the same shape as `data`.

    Args:
        data (numpy.ndarray): Data to mask, must be 2 dimensional array.
        soft_width (int): Width of the soft zone (in pixels).
        len_dim1 (int): Length of the side of the square along first dimension (in pixels).
        len_dim2 (int): Length of the side of the square along second dimension (in pixels).
        center_dim1 (int): Center of the square along first dimension (in pixels). If no center is provided, the middle
                           is used.
        center_dim2 (int): Center of the square along second dimension (in pixels). If no center is provided, the middle
                           is used.
    Returns:
        numpy.ndarray: Mask with floats.
    """
    bin_square_mask = shape_square(data, len_dim1, len_dim2, center_dim1, center_dim2)
    soft_square_mask = np.zeros_like(data)
    for i in range(soft_width):
        dilated_mask = binary_dilation(bin_square_mask, disk(i)).astype(float)
        difference = dilated_mask - soft_square_mask
        soft_square_mask = soft_square_mask + difference * (1 - i/soft_width)
    return soft_square_mask
