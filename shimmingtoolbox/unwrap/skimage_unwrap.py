#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Wrapper to skimage unwrap_phase
"""

import logging
import nibabel as nib
import numpy as np
from numpy import ma
from skimage.restoration import unwrap_phase

from shimmingtoolbox.masking.threshold import threshold as mask_threshold

logger = logging.getLogger(__name__)


def skimage_unwrap(nii_wrapped_phase, mag=None, mask=None, threshold=None, fname_save_mask=None):
    """ Unwraps the phase using skimage unwrap_phase.

    Args:
        nii_wrapped_phase (nib.Nifti1Image): 2D or 3D radian numpy array to perform phase unwrapping. (2 pi interval)
        mag (numpy.ndarray): 2D or 3D magnitude numpy array corresponding to the phase array
        mask (numpy.ndarray): numpy array of booleans with shape of `complex_array` to mask during phase
                                        unwrapping
        threshold: Threshold value for automatic mask generation (Use either mask or threshold, not both)
        fname_save_mask (str): Filename of the mask calculated by the unwrapper

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude
    """

    wrapped_phase = nii_wrapped_phase.get_fdata()
    # Make sure phase and mag are the right shape
    if wrapped_phase.ndim not in [2, 3]:
        raise ValueError("Wrapped_phase must be 2d or 3d")

    # Use the mask or create a mask if a threshold and a mag are provided
    if mask is not None:
        if mask.shape != wrapped_phase.shape:
            raise ValueError("Mask must be the same shape as wrapped_phase")

        unwrap_mask = mask
    elif mag is not None:
        if wrapped_phase.shape != mag.shape:
            raise ValueError("The magnitude image (mag) must be the same shape as wrapped_phase")
        if threshold is not None:
            unwrap_mask = mask_threshold(mag, thr=threshold)
        else:
            raise ValueError("Threshold must be specified if a mag is provided")
    else:
        logger.warning("No mask or mag provided. Unwrapping the whole image, verify for residual wraps.")
        unwrap_mask = np.ones_like(wrapped_phase, dtype=bool)

    # Save the mask if a filename is provided
    if fname_save_mask is not None:
        nii_mask = nib.Nifti1Image(unwrap_mask, nii_wrapped_phase.affine, header=nii_wrapped_phase.header)
        nib.save(nii_mask, fname_save_mask)

    # Unwrap the phase
    ma_wrapped_phase = ma.array(wrapped_phase, mask=~unwrap_mask)
    ma_unwrapped_phase = unwrap_phase(ma_wrapped_phase, rng=0)

    # Fill the masked values with 0
    unwrapped_phase = ma_unwrapped_phase.filled(0)

    return unwrapped_phase
