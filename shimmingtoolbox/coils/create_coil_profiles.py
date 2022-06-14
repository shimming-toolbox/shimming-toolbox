#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


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

    # Define the shape and affine that all inputs should have
    nii = nib.load(fnames_fmaps[0][0])
    affine = nii.affine
    shape = nii.shape

    # Initialize output profiles
    profiles = np.zeros(shape + (n_channels,), dtype=float)

    # Process each channel separately
    for i_channel in range(n_channels):
        n_currents = len(fnames_fmaps[i_channel])
        # Make sure fname_fmaps and list_currents have the same number of currents
        if n_currents != len(list_currents[i_channel]):
            raise ValueError("The number of currents should be the same for the fieldmaps and for the setup currents")

        fmaps = np.zeros(shape + (n_currents,))
        for i_current in range(n_currents):
            nii_fmap = nib.load(fnames_fmaps[i_channel][i_current])

            # Make sure affine ans shape are the same for all channels
            if np.all(nii_fmap.shape != shape):
                raise ValueError("Input shape of fieldmaps must be the same")
            if np.all(nii_fmap.affine != affine):
                raise ValueError("Input affines of fieldmaps must be the same")

            fmaps[..., i_current] = nii_fmap.get_fdata()

        reg = LinearRegression().fit(np.array(list_currents[i_channel]).reshape(-1, 1),
                                     fmaps.reshape(-1, fmaps.shape[-1]).T)

        profiles[..., i_channel] = reg.coef_.reshape(fmaps.shape[:-1])
        # static_offset = reg.intercept_.reshape(fmaps.shape[:-1])

    return profiles
