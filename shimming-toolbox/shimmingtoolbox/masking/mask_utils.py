#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure, iterate_structure, maximum_filter
from skimage.morphology import ball

from shimmingtoolbox.coils.coordinates import resample_from_to

logger = logging.getLogger(__name__)
mask_operations = {'dilate': binary_dilation, 'erode': binary_erosion}


def resample_mask(nii_mask_from, nii_target, from_slices=None, dilation_kernel='None', dilation_size=3,
                  path_output=None, return_non_dil_mask=False):
    """
    Resample a source mask (`nii_mask_from`) onto a target image (`nii_target`) while selecting specific slices
    and applying optional dilation restricted to the region of interest (ROI).

    This function performs the following steps:
    1. **Slice Selection**: If `from_slices` is specified, only the corresponding axial slices from the input mask are used.
    2. **Resampling**: The sliced mask is resampled to match the spatial resolution, dimensions, and orientation of `nii_target`.
    3. **Dilation**: If the mask is binary (i.e., contains only 0/1 or boolean values), a morphological dilation is applied
    using the specified kernel and size. If the mask is soft (i.e., contains float values between 0 and 1), a custom dilation
    is performed using a maximum filter within a spherical neighborhood.
    4. **ROI Constraint**: The dilated mask is intersected with the resampled full original mask (before slice selection)
    to ensure that added voxels remain within the originally defined anatomical region.
    5. **Output**: The function returns the final mask (dilated and ROI-restricted). If `return_non_dil_mask` is True,
    it also returns the undilated mask.

    Args:
        nii_mask_from (nib.Nifti1Image): Source mask to resample. Voxels with value 0 or False are considered outside the mask.
        nii_target (nib.Nifti1Image): Target image defining the desired output space.
        from_slices (tuple): Indices of the slices to select from `nii_mask_from`. If None, all slices are used.
        dilation_kernel (str): Shape of the kernel used for dilation. Allowed shapes: 'sphere', 'cross', 'line', 'cube'.
                            See :func:`modify_binary_mask` for more details.
        dilation_size (int): Size of the 3D dilation kernel. Must be odd. For instance, a size of 3 dilates the mask by 1 voxel.
        path_output (str): Optional path to save masks when debugging.
        return_non_dil_mask (bool): If True, both the dilated and undilated resampled masks are returned.

    Returns:
        nib.Nifti1Image: Mask resampled with nii_target.shape and nii_target.affine.
    """
    mask_from = nii_mask_from.get_fdata()

    if from_slices is None:
        from_slices = tuple(range(mask_from.shape[2]))

    # Initialize a sliced mask and select the slices from from_slices
    sliced_mask = np.full_like(mask_from, fill_value=0)
    sliced_mask[:, :, from_slices] = mask_from[:, :, from_slices]

    # Create nibabel object of sliced mask
    nii_mask = nib.Nifti1Image(sliced_mask.astype(float), nii_mask_from.affine, header=nii_mask_from.header)
    # Resample the sliced mask onto nii_target
    nii_mask_target = resample_from_to(nii_mask, nii_target, order=1, mode='grid-constant', cval=0)
    # Resample the full mask onto nii_target
    nii_full_mask_target = resample_from_to(nii_mask_from, nii_target, order=0, mode='grid-constant', cval=0)

    # Dilate the mask to add more pixels in particular directions
    if np.array_equal(np.unique(nii_mask_from.get_fdata()), [0, 1]) or nii_mask_from.get_fdata().dtype == bool:
        mask_dilated = modify_binary_mask(nii_mask_target.get_fdata(), dilation_kernel, dilation_size, 'dilate')
    else :
        previous_mask = np.array(nii_mask_target.get_fdata(), dtype=float)
        kernel = ball(1)
        for _ in range(dilation_size//2):
            max_filter = maximum_filter(previous_mask, footprint=kernel, mode='grid-constant')
            max_filter[nii_mask_target.get_fdata() != 0] = nii_mask_target.get_fdata()[nii_mask_target.get_fdata() != 0]
            previous_mask = max_filter
        mask_dilated = max_filter

    # Make sure the mask is within the original ROI
    mask_dilated_in_roi = np.zeros_like(mask_dilated)
    mask_dilated_in_roi[nii_full_mask_target.get_fdata() != 0] = mask_dilated[nii_full_mask_target.get_fdata() != 0]
    nii_mask_dilated = nib.Nifti1Image(mask_dilated_in_roi, nii_mask_target.affine, header=nii_mask_target.header)

    # # Save masks for debugging if necessary
    # path_output_original_mask = os.path.join(path_output, "fig_mask_original")
    # os.makedirs(path_output_original_mask, exist_ok=True)
    # path_output_resampled_mask = os.path.join(path_output, "fig_mask_resampled")
    # os.makedirs(path_output_resampled_mask, exist_ok=True)
    # path_output_dilated_mask = os.path.join(path_output, "fig_mask_dilated")
    # os.makedirs(path_output_dilated_mask, exist_ok=True)

    # nib.save(nii_mask, os.path.join(path_output_original_mask, f"fig_mask_original_slice_{from_slices[0]}.nii.gz"))
    # nib.save(nii_mask_target, os.path.join(path_output_resampled_mask, f"fig_mask_resampled_slice_{from_slices[0]}.nii.gz"))
    # nib.save(nii_mask_dilated, os.path.join(path_output_dilated_mask, f"fig_mask_dilated_slice_{from_slices[0]}.nii.gz"))

    # Return non dilated mask if requested
    if return_non_dil_mask:
        mask_in_roi = np.zeros_like(mask_dilated)
        mask_in_roi[nii_full_mask_target.get_fdata() != 0] = nii_mask_target.get_fdata()[nii_full_mask_target.get_fdata() != 0]
        nii_mask_resampled = nib.Nifti1Image(mask_in_roi, nii_mask_target.affine, header=nii_mask_target.header)

        return nii_mask_resampled, nii_mask_dilated

    else:
        return nii_mask_dilated


def modify_binary_mask(mask, shape='sphere', size=3, operation='dilate'):
    """
    Dilates or erodes a binary mask according to different shapes and kernel size

    Args:
        mask (numpy.ndarray): 3d array containing the binary mask.
        shape (str): 3d kernel to perform the dilation. Allowed shapes are: 'sphere', 'cross', 'line', 'cube', 'None'.
                     'line' uses 3 line kernels to extend in each directions by "(size - 1) / 2" only if that direction
                     is smaller than (size - 1) / 2
        size (int): Length of a side of the 3d kernel. Must be odd.
        operation (str): Operation to perform. Allowed operations are: 'dilate', 'erode'.

    Returns:
        numpy.ndarray: Dilated/eroded mask.

    Notes:

        Kernels for

            * 'cross' size 3:
                ::

                      np.array([[[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]],
                                [[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                [[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]])

            * 'sphere' size 5:
                ::

                    np.array([[[0 0 0 0 0],
                               [0 0 0 0 0],
                               [0 0 1 0 0],
                               [0 0 0 0 0],
                               [0 0 0 0 0]],
                              [[0 0 0 0 0],
                               [0 0 1 0 0],
                               [0 1 1 1 0],
                               [0 0 1 0 0],
                               [0 0 0 0 0]],
                              [[0 0 1 0 0],
                               [0 1 1 1 0],
                               [1 1 1 1 1],
                               [0 1 1 1 0],
                               [0 0 1 0 0]],
                              [[0 0 0 0 0],
                               [0 0 1 0 0],
                               [0 1 1 1 0],
                               [0 0 1 0 0],
                               [0 0 0 0 0]],
                              [[0 0 0 0 0],
                               [0 0 0 0 0],
                               [0 0 1 0 0],
                               [0 0 0 0 0],
                               [0 0 0 0 0]]]

            * 'cube' size 3:
                ::

                    np.array([[[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]],
                              [[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]],
                              [[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]])

            * 'line' size 3:
                ::

                  np.array([[[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]],
                            [[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]],
                            [[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]])

    """

    if size % 2 == 0 or size < 3:
        raise ValueError("Size must be odd and greater or equal to 3")

    if operation not in mask_operations:
        raise ValueError(f"Operation <{operation}> not supported. Supported operations are: {list(mask_operations.keys())}")

    # Find the middle pixel, will always work since we check size is odd
    mid_pixel = int((size - 1) / 2)

    if shape == 'sphere':
        # Define kernel to perform the dilation
        struct_sphere_size1 = generate_binary_structure(3, 1)
        struct = iterate_structure(struct_sphere_size1, mid_pixel)

        # Dilate
        mask_dilated = mask_operations[operation](mask, structure=struct)

    elif shape == 'cross':
        struct = np.zeros([size, size, size])
        struct[:, mid_pixel, mid_pixel] = 1
        struct[mid_pixel, :, mid_pixel] = 1
        struct[mid_pixel, mid_pixel, :] = 1

        # Dilate
        mask_dilated = mask_operations[operation](mask, structure=struct)

    elif shape == 'cube':
        # Define kernel to perform the dilation
        struct_cube_size1 = generate_binary_structure(3, 3)
        struct = iterate_structure(struct_cube_size1, mid_pixel)

        # Dilate
        mask_dilated = mask_operations[operation](mask, structure=struct)

    elif shape == 'line':

        struct_dim1 = np.zeros([size, size, size])
        struct_dim1[:, mid_pixel, mid_pixel] = 1
        # Finds where the structure fits
        open1 = binary_opening(mask, structure=struct_dim1)
        # Select Everything that does not fit within the structure and erode along a dim
        dim1 = mask_operations[operation](np.logical_and(np.logical_not(open1), mask), structure=struct_dim1)

        struct_dim2 = np.zeros([size, size, size])
        struct_dim2[mid_pixel, :, mid_pixel] = 1
        # Finds where the structure fits
        open2 = binary_opening(mask, structure=struct_dim2)
        # Select Everything that does not fit within the structure and erode along a dim
        dim2 = mask_operations[operation](np.logical_and(np.logical_not(open2), mask), structure=struct_dim2)

        struct_dim3 = np.zeros([size, size, size])
        struct_dim3[mid_pixel, mid_pixel, :] = 1
        # Finds where the structure fits
        open3 = binary_opening(mask, structure=struct_dim3)
        # Select Everything that does not fit within the structure and erode along a dim
        dim3 = mask_operations[operation](np.logical_and(np.logical_not(open3), mask), structure=struct_dim3)

        mask_dilated = np.logical_or(np.logical_or(np.logical_or(dim1, dim2), dim3), mask)

    elif shape == 'None':
        mask_dilated = mask

    else:
        raise ValueError("Use of non supported algorithm for dilating the mask")

    return mask_dilated
