#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure, iterate_structure
from skimage.morphology import disk

from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.coils.coordinates import resample_from_to

logger = logging.getLogger(__name__)
mask_operations = {'dilate': binary_dilation, 'erode': binary_erosion}


def resample_mask(nii_mask_from, nii_target, from_slices=None, dilation_kernel='None', dilation_size=3,
                  path_output=None, return_non_dil_mask=False):
    """
    Select the appropriate slices from ``nii_mask_from`` using ``from_slices`` and resample onto ``nii_target``

    Args:
        nii_mask_from (nib.Nifti1Image): Mask to resample from. False or 0 signifies not included.
        nii_target (nib.Nifti1Image): Target image to resample onto.
        from_slices (tuple): Tuple containing the slices to select from nii_mask_from. None selects all the slices.
        dilation_kernel (str): kernel used to dilate the mask. Allowed shapes are: 'sphere', 'cross', 'line'
                               'cube'. See :func:`modify_binary_mask` for more details.
        dilation_size (int): Length of a side of the 3d kernel to dilate the mask. Must be odd. For example,
                                         a kernel of size 3 will dilate the mask by 1 pixel.
        path_output (str): Path to output debug artefacts.
        return_non_dil_mask (bool): See if we want to return the dilated and non dilated resampled mask

    Returns:
        nib.Nifti1Image: Mask resampled with nii_target.shape and nii_target.affine.
    """
    mask_from = nii_mask_from.get_fdata()

    if from_slices is None:
        from_slices = tuple(range(mask_from.shape[2]))

    # Initialize a sliced mask and select the slices from from_slices
    sliced_mask = np.full_like(mask_from, fill_value=False)
    sliced_mask[:, :, from_slices] = mask_from[:, :, from_slices]

    # Create nibabel object of sliced mask
    nii_mask = nib.Nifti1Image(sliced_mask.astype(float), nii_mask_from.affine, header=nii_mask_from.header)
    # Resample the sliced mask onto nii_target
    nii_mask_target = resample_from_to(nii_mask, nii_target, order=1, mode='grid-constant', cval=0)
    # Resample the full mask onto nii_target
    nii_full_mask_target = resample_from_to(nii_mask_from, nii_target, order=0, mode='grid-constant', cval=0)
    # TODO: Deal with soft mask
    # Find highest value and stretch to 1
    # Look into dilation of soft mask

    # Dilate the mask to add more pixels in particular directions
    mask_dilated = modify_binary_mask(nii_mask_target.get_fdata(), dilation_kernel, dilation_size, 'dilate')
    # Make sure the mask is within the original ROI
    mask_dilated_in_roi = np.logical_and(mask_dilated, nii_full_mask_target.get_fdata())
    nii_mask_dilated = nib.Nifti1Image(mask_dilated_in_roi, nii_mask_target.affine, header=nii_mask_target.header)

    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_mask, os.path.join(path_output, f"fig_mask_{from_slices[0]}.nii.gz"))
    #     nib.save(nii_mask_target, os.path.join(path_output, f"fig_mask_res{from_slices[0]}.nii.gz"))
    #     nib.save(nii_mask_dilated, os.path.join(path_output, f"fig_mask_dilated{from_slices[0]}.nii.gz"))

    if return_non_dil_mask:
        mask_in_roi = np.logical_and(nii_mask_target.get_fdata(), nii_full_mask_target.get_fdata())
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
                               [0 0  1 0 0],
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


def basic_soft_square_mask(data, soft_width, soft_value, len_dim1, len_dim2, center_dim1=None, center_dim2=None) :
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


def soft_square_mask(data, soft_width, len_dim1, len_dim2, center_dim1=None, center_dim2=None) :
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


def basic_sct_soft_mask(path_sct_bin_mask, path_sct_soft_mask, soft_width, soft_value):
    """
    Creates a basic softmask from a binary mask created from the `sct_create_mask` function.

    Args:
        path_sct_bin_mask (str): Path to the binary mask created from the `sct_create_mask` function.
        path_sct_soft_mask (str): Path to save the soft mask
        soft_width (int): Width of the soft zone (in pixels).
        soft_value (float): Value of the intensity of the pixels in the soft zone.
    Returns:
        nii_sct_soft_mask : NIFTI file containing the soft mask created from the binary mask.
    """
    # Load the binary mask from a NIFTI file
    nifti_file = nib.load(path_sct_bin_mask)
    sct_bin_mask = nifti_file.get_fdata()

    # Create a np.array soft mask
    sct_soft_mask = np.zeros_like(sct_bin_mask)
    for i in range(sct_bin_mask.shape[2]):
        slice = sct_bin_mask[:, :, i]
        dilated_slice = binary_dilation(slice, disk(soft_width)).astype(float)
        difference = dilated_slice - slice
        sct_soft_mask[:, :, i] = slice + difference * soft_value

    # Save the soft mask to a NIFTI file
    nii_sct_soft_mask = nib.Nifti1Image(sct_soft_mask, nifti_file.affine, header=nifti_file.header)
    nii_sct_soft_mask.set_data_dtype(float)
    nii_sct_soft_mask.to_filename(path_sct_soft_mask)


def gradient_sct_soft_mask() :
    pass # TODO


def gaussian_sct_soft_mask() :
    pass # TODO
