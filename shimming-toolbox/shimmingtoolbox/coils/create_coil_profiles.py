#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)
TOLERANCE = 0.001


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

            # Make sure affine and shape are the same for all channels
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


def get_wire_pattern(pumcinFile):
    """ Transform the pumcin file to a usable format
    Args:
        pumcinFile (np.array): 2D array of shape (n_points, 5)

    Returns:
        list: 1D list of wires (channels) with their coordinates formatted
              in n_segments dictionnaries with start and stop coordinates of each segment
    """
    nPoints = pumcinFile.shape[0]
    wireStartPoint = pumcinFile[0, 1:5]
    iChannel = -1

      # [units: mm]

    wires = []

    for iPoint in range(nPoints):
        if pumcinFile[iPoint, 4] == 0:
            wireStartPoint = pumcinFile[iPoint, 1:4]
            iChannel += 1
            iSegment = 0
            wires.append([])
            wires[iChannel].append({})
            wires[iChannel][iSegment]['start'] = wireStartPoint
        else:
            iSegment += 1
            # Wires[iChannel].append({})
            wires[iChannel][iSegment - 1]['stop'] = pumcinFile[iPoint, 1:4]

            if np.linalg.norm(pumcinFile[iPoint, 1:4] - wireStartPoint) < TOLERANCE:
                nSegmentsPerChannel = iSegment - 1
            else:
                wires[iChannel].append({})
                wires[iChannel][iSegment]['start'] = pumcinFile[iPoint, 1:4]

    return wires


def create_coil_config(name, channels, min_current, max_current, max_sum, units):
    """ Create a coil config file

    Args:
        name (str): Name of the coil
        channels (int): Number of channels in the coil
        min_current (float): Minimum coefficient possible
        max_current (float): Maximum coefficient possible
        max_sum (float): Maximum sum of coefficient possible
        units (str): Units of the coefficients e.g. 'A'

    Returns:
        dict: Coil configuration
    """
    if channels < 1:
        raise ValueError("The number of channels must be at least 1")

    if min_current >= max_current:
        raise ValueError("The minimum current must be smaller than the maximum current")

    # Create coil config file
    config_coil = {
        'name': name,
        'coef_channel_minmax': {'coil': [[min_current, max_current]] * channels},
        'coef_sum_max': max_sum,
        'Units': units
    }

    return config_coil
