#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np


def create_coil_profiles(min_max_fmaps, list_diff):
    """ Create coil profiles from fieldmaps

    Args:
        min_max_fmaps (list): 2d list of filename strings (n_channels x 2) containing the min then max currents. The
                              filenames point to fieldmaps in hertz
        list_diff (list): 1d list (n_channels) of floats of the difference current in amps.
                          (min = -0.25A, max = 0.75A, diff = 1)

    Returns:
        numpy.ndarray: Coil profiles
    """
    n_channels = len(min_max_fmaps)
    # Make sure list_diff and min_max_fmaps have the same number of channels
    if n_channels != len(list_diff):
        raise ValueError("Length of min_max_fmaps should be the same as list_diff")

    # Define the shape and affine that all inputs should have
    nii = nib.load(min_max_fmaps[0][0])
    affine = nii.affine
    shape = nii.shape

    # Initialize output profiles
    profiles = np.zeros(shape + (n_channels,), dtype=float)

    # Process each channel separately
    for i_channel in range(n_channels):
        nii_min_fmap = nib.load(min_max_fmaps[i_channel][0])
        nii_max_fmap = nib.load(min_max_fmaps[i_channel][1])

        # Make sure affine ans shape are the same for all channels
        if np.all(nii_min_fmap.shape != shape) or np.all(nii_max_fmap.shape != shape):
            raise ValueError("Input shape of fieldmaps must be the same")
        if np.all(nii_min_fmap.affine != affine) or np.all(nii_max_fmap.affine != affine):
            raise ValueError("Input affines of fieldmaps must be the same")

        min_fmap = nii_min_fmap.get_fdata()
        max_fmap = nii_max_fmap.get_fdata()

        # Process the profiles
        profiles[..., i_channel] = _create_coil_profile(min_fmap, max_fmap, list_diff[i_channel])

    return profiles


def _create_coil_profile(min_fmap, max_fmap, diff):
    profile = (max_fmap - min_fmap) / diff
    return profile
