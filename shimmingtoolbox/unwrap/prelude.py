#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Wrapper to FSL Prelude (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#PRELUDE_.28phase_unwrapping.29)
"""

import glob
import os
import nibabel as nib
import pathlib
import tempfile
import logging
import numpy as np

from shimmingtoolbox.utils import run_subprocess


def prelude(wrapped_phase, affine, mag=None, mask=None, threshold=None, is_unwrapping_in_2d=False):
    """wrapper to FSL prelude

    This function enables phase unwrapping by calling FSL prelude on the command line. A mask can be provided to mask
    the phase image provided. 2D unwrapping can be turned off. The output path can be specified. The temporary niis
    can optionally be saved.

    Args:
        wrapped_phase (numpy.ndarray): 2D or 3D radian numpy array to perform phase unwrapping. (2 pi interval)
        affine (numpy.ndarray): 2D array containing the transformation coefficients. Can be calculated by using:
            nii = nib.load("nii_path")
            affine = nii.affine
        mag (numpy.ndarray): 2D or 3D magnitude numpy array corresponding to the phase array
        mask (numpy.ndarray, optional): numpy array of booleans with shape of `complex_array` to mask during phase
                                        unwrapping
        threshold: Threshold value for automatic mask generation (Use either mask or threshold, not both)
        is_unwrapping_in_2d (bool, optional): prelude parameter to unwrap slice by slice

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude
    """
    # Make sure phase and mag are the right shape
    if wrapped_phase.ndim not in [2, 3]:
        raise RuntimeError("Wrapped_phase must be 2d or 3d")
    if mag is not None:
        if wrapped_phase.shape != mag.shape:
            raise RuntimeError("The magnitude image (mag) must be the same shape as wrapped_phase")
    else:
        mag = np.zeros_like(wrapped_phase)

    tmp = tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem)
    path_tmp = tmp.name

    # Save phase and mag images
    nii_phase = nib.Nifti1Image(wrapped_phase, affine)
    nib.save(nii_phase, os.path.join(path_tmp, 'rawPhase.nii'))
    nii_mag = nib.Nifti1Image(mag, affine)
    nib.save(nii_mag, os.path.join(path_tmp, 'mag.nii'))

    # Fill options
    options = ''
    if is_unwrapping_in_2d:
        options = ' -s'

    # Add mask data and options if there is a mask provided
    if mask is not None:
        if mask.shape != wrapped_phase.shape:
            raise RuntimeError("Mask must be the same shape as wrapped_phase")
        nii_mask = nib.Nifti1Image(mask, affine)

        options += ' -m '
        options += os.path.join(path_tmp, 'mask.nii')
        nib.save(nii_mask, os.path.join(path_tmp, 'mask.nii'))

    if threshold is not None:
        options += ' -t {}'.format(threshold)
        if mask is not None:
            logging.warning('Specifying both a mask and a threshold is not recommended, results might not be what is '
                            'expected')

    # Unwrap
    unwrap_command = 'prelude -p {} -a {} -o {}{}'.format(os.path.join(path_tmp, 'rawPhase'),
                                                          os.path.join(path_tmp, 'mag'),
                                                          os.path.join(path_tmp, 'rawPhase_unwrapped'), options)
    logging.info('Unwrap with prelude')
    run_subprocess(unwrap_command)

    fname_phase_unwrapped = glob.glob(os.path.join(path_tmp, 'rawPhase_unwrapped*'))[0]

    # When loading fname_phase_unwrapped, if a singleton is on the last dimension in wrapped_phase, it will not appear
    # in the last dimension in phase_unwrapped. To be consistent with the size of the input, the singletons are added
    # back.
    phase_unwrapped = nib.load(fname_phase_unwrapped).get_fdata()
    for _ in range(wrapped_phase.ndim - phase_unwrapped.ndim):
        phase_unwrapped = np.expand_dims(phase_unwrapped, -1)

    return phase_unwrapped
