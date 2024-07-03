#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Image thresholding API
"""
import numpy as np


def threshold(data, thr=30, scaled_thr=False):
    """
    Threshold an image

    Args:
        data (numpy.ndarray): Data to be masked
        thr (float): Value to threshold the data: voxels will be set to zero if their value is equal or less than this
        threshold. For complex data, threshold is applied on the absolute values.
        scaled_thr (bool): Specifies if the threshold is absolute or scaled [0, 1].

    Returns:
        numpy.ndarray: Boolean mask with same dimensions as data
    """

    if np.iscomplexobj(data):
        return abs(data) > thr
    else:
        if scaled_thr:
            if 0 <= thr <= 1:
                thr = (data.max() - data.min()) * thr + data.min()
            else:
                raise ValueError(f"Threshold must range between 0 and 1 when using scaled_thr. Input was: {thr}")
        return data > thr
