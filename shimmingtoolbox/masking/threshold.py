#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image thresholding API
"""


def threshold(data, thr=30, b1map=False):
    """
    Threshold an image

    Args:
        data (numpy.ndarray): Data to be masked
        thr (float): Value to threshold the data: voxels will be set to zero if their value is equal or less than this threshold
        b1map(boolean): Specifies if the data are b1 maps (True) or not. Thresholding will be applied accordingly.

    Returns:
        numpy.ndarray: Boolean mask with same dimensions as data
    """
    if b1map:
        return abs(data) > thr
    else:
        return data > thr
