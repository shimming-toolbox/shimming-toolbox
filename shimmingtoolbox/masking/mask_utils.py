#!/usr/bin/python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_opening
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import iterate_structure

from shimmingtoolbox.coils.coordinates import resample_from_to


def resample_mask(nii_mask_from, nii_target, from_slices):
    """
    Select the appropriate slices from ``nii_mask_from`` using ``from_slices`` and resample onto ``nii_target``

    Args:
        nii_mask_from (nib.Nifti1Image): Mask to resample from. False or 0 signifies not included.
        nii_target (nib.Nifti1Image): Target image to resample onto.
        from_slices (tuple): Tuple containing the slices to select from nii_mask_from.

    Returns:
        nib.Nifti1Image: Mask resampled with nii_target.shape and nii_target.affine.
    """

    mask_from = nii_mask_from.get_fdata()

    # Initialize a sliced mask and select the slices from from_slices
    sliced_mask = np.full_like(mask_from, fill_value=False)
    sliced_mask[:, :, from_slices] = mask_from[:, :, from_slices]

    # Create nibabel object
    nii_mask = nib.Nifti1Image(sliced_mask.astype(int), nii_mask_from.affine, header=nii_mask_from.header)

    # Resample the mask onto nii_target
    nii_mask_target = resample_from_to(nii_mask, nii_target, order=0, mode='grid-constant', cval=0)

    # dilate the mask to add more pixels in particular directions
    mask_dilated = dilate_binary_mask(nii_mask_target.get_fdata(), 'line', 3)
    nii_mask_dilated = nib.Nifti1Image(mask_dilated, nii_mask_target.affine, header=nii_mask_target.header)

    # #######
    # # Debug
    # import os
    # nib.save(nii_mask, os.path.join(os.curdir, f"fig_mask_{from_slices[0]}.nii.gz"))
    # nib.save(nii_mask_from, os.path.join(os.curdir, "fig_mask_roi.nii.gz"))
    # nib.save(nii_mask_target, os.path.join(os.curdir, f"fig_mask_res{from_slices[0]}.nii.gz"))
    # nib.save(nii_mask_dilated, os.path.join(os.curdir, f"fig_mask_dilated{from_slices[0]}.nii.gz"))
    # #######

    return nii_mask_dilated


def dilate_binary_mask(mask, shape='cross', size=3):
    """
    Dilates a binary mask according to different shapes and kernel size

    Args:
        mask (numpy.ndarray): 3d array containing the binary mask.
        shape (str): 3d kernel to perform the dilation. Allowed shapes are: 'sphere', 'cross', 'line', 'cube'.
                     'line' uses 3 line kernels to extend in each directions by "(size - 1) / 2" only if that direction
                     is smaller than (size - 1) / 2
        size (int): length of a side of the 3d kernel.

    Returns:
        numpy.ndarray: Dilated mask.

    Notes:
        Kernels for
            'cross':
                array([[[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]],
                       [[0., 1., 0.],
                        [1., 1., 1.],
                        [0., 1., 0.]],
                       [[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]]])

            'sphere' size 5:
                [[[False False False False False]
                  [False False False False False]
                  [False False  True False False]
                  [False False False False False]
                  [False False False False False]]
                 [[False False False False False]
                  [False False  True False False]
                  [False  True  True  True False]
                  [False False  True False False]
                  [False False False False False]]
                 [[False False  True False False]
                  [False  True  True  True False]
                  [ True  True  True  True  True]
                  [False  True  True  True False]
                  [False False  True False False]]
                 [[False False False False False]
                  [False False  True False False]
                  [False  True  True  True False]
                  [False False  True False False]
                  [False False False False False]]
                 [[False False False False False]
                  [False False False False False]
                  [False False  True False False]
                  [False False False False False]
                  [False False False False False]]]

            'cube':
                array([[[ True,  True,  True],
                        [ True,  True,  True],
                        [ True,  True,  True]],
                       [[ True,  True,  True],
                        [ True,  True,  True],
                        [ True,  True,  True]],
                       [[ True,  True,  True],
                        [ True,  True,  True],
                        [ True,  True,  True]]])

            'line':
                array([[[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]],
                       [[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]],
                       [[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]]])

    """

    if size % 2 == 0 or size < 3:
        raise ValueError("Size must be odd and greater or equal to 3")
    # Find the middle pixel, will always work since we check size is odd
    mid_pixel = int((size - 1) / 2)

    if shape == 'sphere':
        # Define kernel to perform the dilation
        struct_sphere_size1 = generate_binary_structure(3, 1)
        struct = iterate_structure(struct_sphere_size1, mid_pixel)

        # Dilate
        mask_dilated = binary_dilation(mask, structure=struct)

    elif shape == 'cross':
        struct = np.zeros([size, size, size])
        struct[:, mid_pixel, mid_pixel] = 1
        struct[mid_pixel, :, mid_pixel] = 1
        struct[mid_pixel, mid_pixel, :] = 1

        # Dilate
        mask_dilated = binary_dilation(mask, structure=struct)

    elif shape == 'cube':
        # Define kernel to perform the dilation
        struct_cube_size1 = generate_binary_structure(3, 3)
        struct = iterate_structure(struct_cube_size1, mid_pixel)

        # Dilate
        mask_dilated = binary_dilation(mask, structure=struct)

    elif shape == 'line':

        struct_dim1 = np.zeros([size, size, size])
        struct_dim1[:, mid_pixel, mid_pixel] = 1
        # Finds where the structure fits
        open1 = binary_opening(mask, structure=struct_dim1)
        # Select Everything that does not fit within the structure and erode along a dim
        dim1 = binary_dilation(np.logical_and(np.logical_not(open1), mask), structure=struct_dim1)

        struct_dim2 = np.zeros([size, size, size])
        struct_dim2[mid_pixel, :, mid_pixel] = 1
        # Finds where the structure fits
        open2 = binary_opening(mask, structure=struct_dim2)
        # Select Everything that does not fit within the structure and erode along a dim
        dim2 = binary_dilation(np.logical_and(np.logical_not(open2), mask), structure=struct_dim2)

        struct_dim3 = np.zeros([size, size, size])
        struct_dim3[mid_pixel, mid_pixel, :] = 1
        # Finds where the structure fits
        open3 = binary_opening(mask, structure=struct_dim3)
        # Select Everything that does not fit within the structure and erode along a dim
        dim3 = binary_dilation(np.logical_and(np.logical_not(open3), mask), structure=struct_dim3)

        mask_dilated = np.logical_or(np.logical_or(np.logical_or(dim1, dim2), dim3), mask)

    else:
        raise ValueError("Use of non supported algorithm for dilating the mask")

    return mask_dilated.astype(int)
