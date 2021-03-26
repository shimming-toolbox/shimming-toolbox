#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import nibabel as nib
from typing import List

ListNii = List[nib.Nifti1Image]


def concat_data(list_nii: ListNii, axis, pixdim=None):
    """
    Concatenate data

    Args:
        list_nii: list of Nifti1Image
        axis: axis: 0, 1, 2, 3, 4.
        pixdim: pixel resolution to join to image header
    Returns:
        ListNii: concatenated image
    """

    dat_list = []
    data_concat_list = []

    for i, nii_im in enumerate(list_nii):
        # if there is more than 100 images to concatenate, then it does it iteratively to avoid memory issue.
        if i != 0 and i % 100 == 0:
            data_concat_list.append(np.concatenate(dat_list, axis=axis))
            dat = nii_im.get_fdata()
            # if image shape is smaller than asked dim, then expand dim
            if len(nii_im.shape) <= axis:
                dat = _expand_dims(dat, axis)
            dat_list = [dat]
            del nii_im
            del dat
        else:
            dat = nii_im.get_fdata()
            # if image shape is smaller than asked dim, then expand dim
            if len(nii_im.shape) <= axis:
                dat = _expand_dims(dat, axis)
            dat_list.append(dat)
            del nii_im
            del dat
    if data_concat_list:
        data_concat_list.append(np.concatenate(dat_list, axis=axis))
        data_concat = np.concatenate(data_concat_list, axis=axis)
    else:
        data_concat = np.concatenate(dat_list, axis=axis)

    im_in_first = list_nii[0]
    nii_out = nib.Nifti1Image(data_concat, im_in_first.affine, im_in_first.header)

    if pixdim is not None:
        cur_pixdim = list_nii[0].header['pixdim']
        cur_pixdim[axis + 1] = pixdim
        nii_out.header['pixdim'] = cur_pixdim

    return nii_out


def _expand_dims(data, axis):
    """
    Expand the shape of an array.

    Wrapper to np.expand_dims allowing axis to be any dimension greater than 0

    Args:
        data (numpy.ndarray): Input array.
        axis (int): axis

    Returns:
        numpy.ndarray: Expanded array

    """

    while len(data.shape) <= axis:
        data = np.expand_dims(data, len(data.shape))

    return data
