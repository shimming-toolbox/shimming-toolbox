#!/usr/bin/python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.morphology import ball


def create_softmasks(fname_binmask, fname_softmask=None, type='2levels', soft_width=6, soft_units='mm', soft_value=0.5):
    """
    Create a soft mask from a binary mask by adding a soft zone around the binary mask.

    Args:
        fname_binmask (str): Path to the binary mask.
        fname_softmask (str): Path to an existing soft mask. Used only if type is 'sum'.
        type (str): Type of soft mask to create. Allowed types are: '2levels', 'linear', 'gaussian', 'sum'.
        soft_width (float): Width of the soft zone.
        soft_units (str): Units of the soft width ('mm' or 'px').
        soft_value (float): Value of the intensity of the pixels in the soft zone. Used only if type is '2levels'.

    Returns:
        numpy.ndarray: 3D array containing the soft mask.
    """
    nifti_binmask = nib.load(fname_binmask)
    binmask = nifti_binmask.get_fdata()
    softmask = None
    if fname_softmask is not None:
        nifti_softmask = nib.load(fname_softmask)
        softmask = nifti_softmask.get_fdata()

    soft_width_px = convert_to_pixels(soft_width, soft_units, nifti_binmask.header)
        
    if type == '2levels':
        return create_two_levels_softmask(binmask, soft_width_px, soft_value)
    elif type == 'linear':
        return create_linear_softmask(binmask, soft_width_px)
    elif type == 'gaussian':
        return create_gaussian_softmask(binmask, soft_width_px)
    if type == 'sum':
        return add_softmask_to_binmask(binmask, softmask)
    else:
        raise ValueError("Invalid soft mask type. Must be one of: '2levels', 'linear', 'gaussian', 'sum'")


def convert_to_pixels(lenght, units, header):
    """
    Convert a lenght from mm to pixels based on voxel size.

    Args:
        lenght (float): Lenght in mm or pixels.
        units (str): Units of the lenght ('mm' or 'px').
        header (nib.Nifti1Header): NIFTI header containing voxel size information.

    Returns:
        int: Blur lenght in pixels."""
    if units == 'mm':
        voxel_sizes = header.get_zooms()
        return int(round(float(lenght) / min(voxel_sizes)))
    elif units == 'px':
        return int(lenght)
    else:
        raise ValueError("Lenght must be 'mm' or 'px'")


def create_two_levels_softmask(binary_mask, soft_width, soft_value):
    """
    Creates a soft mask from a binary mask. The final mask combines the binary mask and its dilated version
    multiplied by a soft value.

    Args:
        binary_mask (numpy.ndarray): 3D array containing the binary mask.
        soft_width (int): Width of the soft zone (in pixels).
        soft_value (float): Value of the intensity of the pixels in the soft zone.

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
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


def create_linear_softmask(binary_mask, soft_width):
    """
    Creates a soft mask from a binary mask. The final mask contains a linear gradient from the binary mask to
    the background.

    Args:
        binary_mask (numpy.ndarray): 3D array containing the binary mask.
        soft_width (int): Width of the soft zone (in pixels).

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
    soft_mask = np.array(binary_mask, dtype=float)
    previous_mask = np.array(binary_mask, dtype=float)

    for i in range(1, soft_width + 1):
        dilated_mask = binary_dilation(previous_mask, structure=ball(1))
        new_layer = dilated_mask & ~previous_mask.astype(bool)
        soft_mask[new_layer] = 1 - (i / (soft_width + 1))
        previous_mask = dilated_mask

    soft_mask = np.clip(soft_mask, 0, 1)

    return soft_mask


def create_gaussian_softmask(binary_mask, soft_width):
    """
    Creates a soft mask from a binary mask. The final mask contains a gaussian blur from the binary mask to
    the background.

    Args:
        binary_mask (numpy.ndarray): 3D array containing the binary mask.
        soft_width (int): Width of the soft zone (in pixels).

    Returns:
        numpy.ndarray: Soft mask created from the binary mask.
    """
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

