#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import nibabel as nib
import numpy as np
# import os

from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure, iterate_structure, gaussian_filter, maximum_filter
from skimage.morphology import ball

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
    sliced_mask = np.full_like(mask_from, fill_value=0)
    sliced_mask[:, :, from_slices] = mask_from[:, :, from_slices]

    # Create nibabel object of sliced mask
    nii_mask = nib.Nifti1Image(sliced_mask.astype(float), nii_mask_from.affine, header=nii_mask_from.header)
    # Resample the sliced mask onto nii_target
    nii_mask_target = resample_from_to(nii_mask, nii_target, order=0, mode='grid-constant', cval=0)
    # Resample the full mask onto nii_target
    nii_full_mask_target = resample_from_to(nii_mask_from, nii_target, order=0, mode='grid-constant', cval=0)

    # 2 options:
    # - max_filter
    # - mean

    # mean_filter
    # kernel = np.array([[[0, 0, 0],
    #                              [0, 1, 0],
    #                              [0, 0, 0]],
    #                             [[0, 1, 0],
    #                              [1, 1, 1],
    #                              [0, 1, 0]],
    #                             [[0, 0, 0],
    #                              [0, 1, 0],
    #                              [0, 0, 0]]]) / 3
    # mean_filter = convolve(nii_mask_target.get_fdata(), kernel, mode='grid-constant')
    # mean_filter[nii_mask_target.get_fdata() != 0] = nii_mask_target.get_fdata()[nii_mask_target.get_fdata() != 0]
    # nii_sliced_mask_convolve = nib.Nifti1Image(mean_filter, nii_mask_target.affine, header=nii_mask_target.header)

    # max_filter
    kernel = np.array([[[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]],
                                [[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                [[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]])
    max_filter = maximum_filter(nii_mask_target.get_fdata(), footprint=kernel, mode='grid-constant')
    max_filter[nii_mask_target.get_fdata() != 0] = nii_mask_target.get_fdata()[nii_mask_target.get_fdata() != 0]
    # nii_sliced_mask_max_filter = nib.Nifti1Image(max_filter, nii_mask_target.affine, header=nii_mask_target.header)

    # path_output = "/Users/antoineguenette/Downloads/"
    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_sliced_mask_max_filter, os.path.join(path_output, "mask_res_on_fmap_slice_max_filter.nii.gz"))

    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_mask_target, os.path.join(path_output, "mask_res_on_fmap_slice.nii.gz"))

    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_full_mask_target, os.path.join(path_output, "mask_res_on_fmap_full.nii.gz"))

    # Dilate the mask to add more pixels in particular directions
    # mask_dilated = modify_binary_mask(nii_mask_target.get_fdata(), dilation_kernel, dilation_size, 'dilate')
    mask_dilated = max_filter

    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nii_sliced_mask_dilated = nib.Nifti1Image(mask_dilated, nii_mask_target.affine, header=nii_mask_target.header)
    #     nib.save(nii_sliced_mask_dilated, os.path.join(path_output, "mask_res_on_fmap_dilated.nii.gz"))

    # Make sure the mask is within the original ROI
    max_filter[nii_mask_target.get_fdata() != 0] = nii_mask_target.get_fdata()[nii_mask_target.get_fdata() != 0]
    mask_dilated_in_roi = np.zeros_like(mask_dilated)
    mask_dilated_in_roi[nii_full_mask_target.get_fdata() != 0] = mask_dilated[nii_full_mask_target.get_fdata() != 0]

    # mask_dilated_in_roi = np.logical_and(mask_dilated, nii_full_mask_target.get_fdata())
    nii_mask_dilated = nib.Nifti1Image(mask_dilated_in_roi, nii_mask_target.affine, header=nii_mask_target.header)
    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_mask_dilated, os.path.join(path_output, "mask_dilated_restricted.nii.gz"))

    # if logger.level <= getattr(logging, 'DEBUG') and path_output is not None:
    #     nib.save(nii_mask, os.path.join(path_output, f"fig_mask_{from_slices[0]}.nii.gz"))
    #     nib.save(nii_mask_target, os.path.join(path_output, f"fig_mask_res{from_slices[0]}.nii.gz"))
    #     nib.save(nii_mask_dilated, os.path.join(path_output, f"fig_mask_dilated{from_slices[0]}.nii.gz"))

    if return_non_dil_mask:
        # TODO: Probably not important to do the logical and?
        # mask_in_roi = np.logical_and(nii_mask_target.get_fdata(), nii_full_mask_target.get_fdata())
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


def create_2vals_softmask(path_binmask, soft_width, soft_value):
    """
    Creates a soft mask from a binary mask. The final mask combines the binary mask and a dilated version muliplied by
    a soft value.

    Args:
        path_sct_binmask (str): Path to the binary mask.
        soft_width (int): Width of the soft zone (in pixels). Must be a multiple of 3.
        soft_value (float): Value of the intensity of the pixels in the soft zone.
    Returns:
        numpy.ndarray : Soft mask created from the binary mask.
    """
    # Load the binary mask from a NIFTI file
    nifti_file = nib.load(path_binmask)
    binmask = nifti_file.get_fdata()

    # Raise error if soft_width is not a multiple of 3
    if soft_width % 3 != 0:
        raise ValueError("soft_width must be a multiple of 3")

    # Create a soft mask
    softmask = np.array(binmask, dtype=float)
    previous_mask = np.array(binmask, dtype=float)
    for _ in range(soft_width // 3):
        dilated_mask = binary_dilation(previous_mask, ball(3))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        softmask[new_layer] = soft_value
        previous_mask = dilated_mask
    softmask = np.clip(softmask, 0, 1)

    return softmask


def create_linear_softmask(path_binmask, soft_width):
    """
    Creates a soft mask from a binary mask. The final mask contains a linear gradient from the binary mask to
    the background.

    Args:
        path_sct_binmask (str): Path to the binary mask.
        soft_width (int): Width of the soft zone (in pixels).
    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    # Load the binary mask from a NIFTI file
    nifti_file = nib.load(path_binmask)
    binmask = nifti_file.get_fdata()

    # Create a np.array soft mask
    softmask = np.array(binmask, dtype=float)
    previous_mask = np.array(binmask, dtype=float)
    for i in range(1, soft_width + 1):
        dilated_mask = binary_dilation(previous_mask, structure=ball(1))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        softmask[new_layer] = 1 - (i / (soft_width + 1))
        previous_mask = dilated_mask
    softmask = np.clip(softmask, 0, 1)

    return softmask


def create_gaussian_softmask(path_binmask, soft_width):
    """
    Creates a softmask from a binary mask. The final mask contains a gaussian blur from the binary mask to
    the background.

    Args:
        path_sct_binmask (str): Path to the binary mask.
        soft_width (int): Width of the soft zone (in pixels). Must be a multiple of 3.
    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    # Load the binary mask from a NIFTI file
    nifti_file = nib.load(path_binmask)
    binmask = nifti_file.get_fdata()

    # Raise error if soft_width is not a multiple of 3
    if soft_width % 3 != 0:
        raise ValueError("soft_width must be a multiple of 3")

    # Create a np.array soft mask
    softmask = np.array(binmask, dtype=float)
    previous_mask = np.array(binmask, dtype=float)
    for _ in range(soft_width // 3):
        dilated_mask = binary_dilation(previous_mask, ball(3))
        previous_mask = dilated_mask
    blurred_mask = gaussian_filter(previous_mask.astype(float), soft_width)
    softmask = np.clip(blurred_mask + binmask, 0, 1)

    # Crop the soft mask to the soft width
    softmask[previous_mask == 0] = 0

    return softmask


def gaussian_sct_softmask(path_sct_binmask, path_sct_gaussmask):
    """
    Adapts the gaussian filter created by sct_create_mask to create a soft mask containing a gaussian blur from the
    binary mask to the background, while keeping the binary mask.

    Args:
        path_sct_binmask (str): Path to the binary mask created from the `sct_create_mask` function.
        path_sct_gaussmask (str): Path to the gaussian mask created from the `sct_create_mask` function.
    Returns:
        sct_softmask (numpy.ndarray): soft mask created from the SCT masks.
    """
    # Load the sct binary mask from a NIFTI file
    nifti_file = nib.load(path_sct_binmask)
    sct_binmask = nifti_file.get_fdata()

    # Load the sct gaussian mask from a NIFTI file
    nifti_file = nib.load(path_sct_gaussmask)
    sct_gaussmask = nifti_file.get_fdata()

    # Create a np.array soft mask
    sct_gaussmask[sct_gaussmask < 0.1] = 0
    sct_softmask = np.clip(sct_gaussmask + sct_binmask, 0, 1)

    return sct_softmask


def save_softmask(softmask, path_softmask, path_binmask):
    """
    Save the soft mask to a NIFTI file

    Args:
        softmask (numpy.ndarray): Soft mask to save.
        path_softmask (str): Path to save the soft mask.
        path_binmask (str): Path to the binary mask.
    Returns:
        nib.Nifti1Image : NIFTI file containing the soft mask created from the binary mask.
    """
    nifti_file = nib.load(path_binmask)

    nii_softmask = nib.Nifti1Image(softmask, nifti_file.affine)
    nii_softmask.set_data_dtype(float)
    nii_softmask.to_filename(path_softmask)

    return nii_softmask
