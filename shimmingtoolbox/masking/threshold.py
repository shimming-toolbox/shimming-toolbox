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
        thr: Value to threshold the data: voxels will be set to zero if their value is equal or less than this threshold

    Returns:
        numpy.ndarray: Boolean mask with same dimensions as data
    """
    return abs(data) > thr
