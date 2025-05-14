#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import nibabel as nib
import numpy as np

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

    # Dilate the mask to add more pixels in particular directions
    if np.array_equal(np.unique(nii_mask_target.get_fdata()), [0, 1]) or nii_mask_target.get_fdata().dtype == bool:
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


def create_two_levels_softmask(nii_binary_mask, soft_width, soft_value, soft_width_units='mm'):
    """
    Creates a soft mask from a binary mask. The final mask combines the binary mask and its dilated version
    multiplied by a soft value.

    Args:
        nii_binary_mask (nib.Nifti1Image): Nifti image object containing the binary mask.
        soft_width (Union[int, str]): Width of the soft zone.
        soft_value (float): Value of the intensity of the pixels in the soft zone.
        soft_width_units (str): Units of the soft width. Can be 'mm' or 'px'

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    # Get the soft width in pixels
    if soft_width_units == 'mm' :
        voxel_sizes = nii_binary_mask.header.get_zooms()
        max_voxel_size = min(voxel_sizes)
        soft_width = int(round(float(soft_width)/ max_voxel_size))
    elif soft_width_units == 'px':
        soft_width = int(soft_width)
    else:
        raise ValueError("soft_width_units must be 'mm' or 'voxels'")

    # Create the soft mask
    binary_mask = nii_binary_mask.get_fdata()
    soft_mask = np.array(binary_mask, dtype=float)
    previous_mask = np.array(binary_mask, dtype=float)

    for _ in range(soft_width // 3):
        dilated_mask = binary_dilation(previous_mask, ball(3))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        soft_mask[new_layer] = soft_value
        previous_mask = dilated_mask

    remainder = soft_width % 3
    if remainder > 0:
        dilated_mask = binary_dilation(previous_mask, ball(remainder))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        soft_mask[new_layer] = soft_value

    soft_mask = np.clip(soft_mask, 0, 1)

    return soft_mask


def create_linear_softmask(nii_binary_mask, soft_width, soft_width_units='mm'):
    """
    Creates a soft mask from a binary mask. The final mask contains a linear gradient from the binary mask to
    the background.

    Args:
        nii_binary_mask (nib.Nifti1Image): Nifti image object containing the binary mask.
        soft_width (Union[int, str]): Width of the soft zone.
        soft_width_units (str): Units of the soft width. Can be 'mm' or 'px'

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    # Get the soft width in pixels
    if soft_width_units == 'mm' :
        voxel_sizes = nii_binary_mask.header.get_zooms()
        max_voxel_size = min(voxel_sizes)
        soft_width = int(round(float(soft_width)/ max_voxel_size))
    elif soft_width_units == 'px':
        soft_width = int(soft_width)
    else:
        raise ValueError("soft_width_units must be 'mm' or 'voxels'")

    # Create the soft mask
    binary_mask = nii_binary_mask.get_fdata()
    soft_mask = np.array(binary_mask, dtype=float)
    previous_mask = np.array(binary_mask, dtype=float)

    for i in range(1, soft_width + 1):
        dilated_mask = binary_dilation(previous_mask, structure=ball(1))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        soft_mask[new_layer] = 1 - (i / (soft_width + 1))
        previous_mask = dilated_mask

    soft_mask = np.clip(soft_mask, 0, 1)

    return soft_mask


def create_gaussian_softmask(nii_binary_mask, soft_width, soft_width_units='mm'):
    """
    Creates a soft mask from a binary mask. The final mask contains a gaussian blur from the binary mask to
    the background.

    Args:
        nii_binary_mask (nib.Nifti1Image): Nifti image object containing the binary mask.
        soft_width (Union[int, str]): Width of the soft zone.
        soft_width_units (str): Units of the soft width. Can be 'mm' or 'px'

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    # Get the soft width in pixels
    if soft_width_units == 'mm' :
        voxel_sizes = nii_binary_mask.header.get_zooms()
        max_voxel_size = min(voxel_sizes)
        soft_width = int(round(float(soft_width)/ max_voxel_size))
    elif soft_width_units == 'px':
        soft_width = int(soft_width)
    else:
        raise ValueError("soft_width_units must be 'mm' or 'voxels'")

    # Create the soft mask
    binary_mask = nii_binary_mask.get_fdata()
    soft_mask = np.array(binary_mask, dtype=float)
    previous_mask = np.array(binary_mask, dtype=float)

    for _ in range(soft_width // 3):
        previous_mask = binary_dilation(previous_mask, ball(3))

    remainder = soft_width % 3
    if remainder > 0:
        previous_mask = binary_dilation(previous_mask, ball(remainder))

    blurred_mask = gaussian_filter(previous_mask.astype(float), soft_width)
    soft_mask = np.clip(blurred_mask + binary_mask, 0, 1)
    soft_mask[previous_mask == 0] = 0

    return soft_mask


def add_softmask_to_binmask(soft_mask, binary_mask):
    """
    Adds a soft mask to a binary mask to create a new soft mask.

    Args:
        soft_mask (numpy.ndarray): 3D array containing the soft mask.
        binary_mask (numpy.ndarray): 3D array containing the binary mask.

    Returns:
        numpy.ndarray: New soft mask.
    """
    soft_mask = np.clip(soft_mask + binary_mask, 0, 1)

    return soft_mask


def save_softmask(soft_mask, fname_soft_mask, fname_binary_mask):
    """
    Save the soft mask to a NIFTI file

    Args:
        soft_mask (numpy.ndarray): 3D array containing the soft mask.
        fname_soft_mask (str): Path to save the soft mask.
        fname_binary_mask (str): Path to the binary mask used to create the soft mask.

    Returns:
        nib.Nifti1Image : NIFTI file containing the soft mask created from the binary mask.
    """
    nii_binary_mask = nib.load(fname_binary_mask)

    nii_softmask = nib.Nifti1Image(soft_mask, nii_binary_mask.affine)
    nii_softmask.set_data_dtype(float)
    nii_softmask.to_filename(fname_soft_mask)

    return nii_softmask
