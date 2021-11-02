#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image thresholding API
"""


def threshold(data, thr=30):
    """
    Threshold an image

    Args:
        data (numpy.ndarray): Data to be masked
        thr (float): Value to threshold the data: voxels will be set to zero if their value is equal or less than this
        threshold. For complex data, threshold is applied on the absolute values.

    Returns:
        numpy.ndarray: Boolean mask with same dimensions as data
    """
    if data.dtype == 'complex128':
        return abs(data) > thr
    else:
        return data > thr
