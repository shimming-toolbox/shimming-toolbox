#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np


def create_coil_profiles(fnames_fmaps, list_currents):
    """ Create coil profiles from fieldmaps

    Args:
        fnames_fmaps (list): 2d list of filename strings (n_channels x n_currents) containing the currents. The
                              filenames point to fieldmaps in hertz.
        list_currents (list): 2d list (n_channels x n_currents) of floats of the currents in amps. For a 2 channel coil
                              acquired at -0.25A and 0.75A, it would have: [[-0.25, 0.75], [-0.25, 0.75]]

    Returns:
        numpy.ndarray: Coil profiles
    """
    n_channels = len(fnames_fmaps)
    # Make sure list_currents and fnames_fmaps have the same number of channels
    if n_channels != len(list_currents):
        raise ValueError("The number of channels should be the same for the fieldmaps and for the setup currents")

    n_currents = len(fnames_fmaps[0])
    # Make sure fname_fmaps and list_currents have the same number of currents
    if n_currents != len(list_currents[0]):
        raise ValueError("The number of currents should be the same for the fieldmaps and for the setup currents")

    # Define the shape and affine that all inputs should have
    nii = nib.load(fnames_fmaps[0][0])
    affine = nii.affine
    shape = nii.shape

    # Initialize output profiles
    profiles = np.zeros(shape + (n_channels,), dtype=float)

    # Process each channel separately
    for i_channel in range(n_channels):

        if n_currents == 2:

            nii_min_fmap = nib.load(fnames_fmaps[i_channel][0])
            nii_max_fmap = nib.load(fnames_fmaps[i_channel][1])

            # Make sure affine ans shape are the same for all channels
            if np.all(nii_min_fmap.shape != shape) or np.all(nii_max_fmap.shape != shape):
                raise ValueError("Input shape of fieldmaps must be the same")
            if np.all(nii_min_fmap.affine != affine) or np.all(nii_max_fmap.affine != affine):
                raise ValueError("Input affines of fieldmaps must be the same")

            min_fmap = nii_min_fmap.get_fdata()
            max_fmap = nii_max_fmap.get_fdata()

            # Process the profiles
            diff = list_currents[i_channel][1] - list_currents[i_channel][0]
            profiles[..., i_channel] = _create_coil_profile(min_fmap, max_fmap, diff)

        else:
            # TODO: Implement coil profile generation for more than 2 currents
            raise NotImplementedError("Only supports 2 different currents")

    return profiles


def _create_coil_profile(min_fmap, max_fmap, diff):
    profile = (max_fmap - min_fmap) / diff
    return profile
