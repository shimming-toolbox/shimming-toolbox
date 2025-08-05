#!/usr/bin/python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

def create_softmask(fname_binmask, fname_softmask=None, type='2levels', soft_width=6, width_unit='mm', soft_value=0.5):
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
    # Load masks
    nii_binmask = nib.load(fname_binmask)
    binmask = nii_binmask.get_fdata()
    softmask = None
    if fname_softmask is not None:
        nii_softmask = nib.load(fname_softmask)
        softmask = nii_softmask.get_fdata()

    # Convert mm to px if needed
    if width_unit == 'mm':
        voxel_sizes = nii_binmask.header.get_zooms()
        soft_width_px = int(round(float(soft_width) / min(voxel_sizes)))
    elif width_unit == 'px':
        soft_width_px = int(soft_width)
    else :
        raise ValueError("Lenght must be 'mm' or 'px'")

    soft_mask_options = {
    '2levels': lambda: create_two_levels_softmask(binmask, soft_width_px, soft_value),
    'linear': lambda: create_linear_softmask(binmask, soft_width_px),
    'gaussian': lambda: create_gaussian_softmask(binmask, soft_width_px),
    'sum': lambda: add_softmask_to_binmask(binmask, softmask)
}
    try:
        return soft_mask_options[type]()
    except KeyError:
        raise ValueError("Invalid soft mask type. Must be one of: soft_mask_options.keys()")


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

    # Invert mask: inside object is 0, outside is 1
    outside_mask = ~binary_mask.astype(bool)

    # Compute distance for outside voxels to nearest inside
    dist = distance_transform_edt(outside_mask)

    # Create soft region: voxels within soft_width from the edge
    soft_region = (dist > 0) & (dist <= soft_width)

    # Apply soft values to soft_mask
    soft_mask[soft_region] = soft_value
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

    # Invert mask: inside object is 0, outside is 1
    outside_mask = ~binary_mask.astype(bool)

    # Compute distance for outside voxels to nearest inside
    dist = distance_transform_edt(outside_mask)

    # Create soft region: voxels within soft_width from the edge
    soft_region = (dist > 0) & (dist <= soft_width)

    # Compute linear weights in soft region: value = 1 - dist / (soft_width + 1)
    # Ensures value = 1 - 1 / (soft_width + 1) at dist=0, and value = 0 at dist=soft_width + 1
    soft_values = 1 - dist[soft_region] / (soft_width + 1)

    # Apply soft values to soft_mask
    soft_mask[soft_region] = soft_values
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

    # Invert mask: inside object is 0, outside is 1
    outside_mask = ~binary_mask.astype(bool)

    # Compute distance for outside voxels to nearest inside
    dist = distance_transform_edt(outside_mask)

    # Create soft region: voxels within soft_width from the edge
    soft_region = (dist > 0) & (dist <= soft_width)

    # Gaussian decay: value = exp(-dist^2 / (2 * sigma^2))
    sigma = soft_width / 3  # ~99.7% of Gaussian within 3σ
    soft_values = np.exp(-(dist[soft_region]**2) / (2 * sigma**2))

    # Apply soft values to soft_mask
    soft_mask[soft_region] = soft_values
    soft_mask = np.clip(soft_mask, 0, 1)

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
