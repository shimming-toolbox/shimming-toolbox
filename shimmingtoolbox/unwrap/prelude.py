#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Wrapper to FSL Prelude (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#PRELUDE_.28phase_unwrapping.29)
"""

import glob
import os
import nibabel as nib
import pathlib
import subprocess
import tempfile


def prelude(wrapped_phase, mag, affine, mask=None, is_unwrapping_in_2d=True, is_saving_nii=False):
    """wrapper to FSL prelude

    This function enables phase unwrapping by calling FSL prelude on the command line. A mask can be provided to mask
    the phase image provided. 2D unwrapping can be turned off. The output path can be specified. The temporary niis
    can optionally be saved.

    Args:
        wrapped_phase (numpy.ndarray): 3D radian numpy array to perform phase unwrapping. (2 pi interval)
        mag (numpy.ndarray): 3D magnitude numpy array corresponding to the phase array
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be calculated by using:
            nii = nib.load("nii_path")
            affine = nii.affine
        mask (numpy.ndarray, optional): numpy array of booleans with shape of `complex_array` to mask during phase
                                        unwrapping
        is_unwrapping_in_2d (bool, optional): prelude parameter to unwrap in 2d
        is_saving_nii (bool, optional): specify whether `wrapped_phase`, `mag`, `affine`, `mask` and `unwrapped_phase`
                                        nii files will be saved

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude
    """
    # Make sure phase and mag are the right shape
    if wrapped_phase.ndim != 3:
        raise RuntimeError('wrapped_phase must be 3d')
    if wrapped_phase.shape != mag.shape:
        raise RuntimeError('The magnitude image (mag) must be the same shape as wrapped_phase')

    tmp = tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem)
    path_tmp = tmp.name

    # Save phase and mag images
    phase_nii = nib.Nifti1Image(wrapped_phase, affine)
    nib.save(phase_nii, os.path.join(path_tmp, 'rawPhase.nii'))
    mag_nii = nib.Nifti1Image(mag, affine)
    nib.save(mag_nii, os.path.join(path_tmp, 'mag.nii'))

    # Fill options
    options = ' '
    if is_unwrapping_in_2d:
        options = '-s '

    # Add mask data and options if there is a mask provided
    if mask is not None:
        if mask.shape != wrapped_phase.shape:
            raise RuntimeError('Mask must be the same shape as wrapped_phase')
        mask_nii = nib.Nifti1Image(mask, affine)

        options += '-m '
        options += os.path.join(path_tmp.name, 'mask.nii')
        nib.save(mask_nii, os.path.join(path_tmp, 'mask.nii'))

    # Unwrap
    unwrap_command = 'prelude -p {} -a {} -o {} {}'.format(os.path.join(path_tmp, 'rawPhase'),
                                                           os.path.join(path_tmp, 'mag'),
                                                           os.path.join(path_tmp, 'rawPhase_unwrapped'), options)
    subprocess.run(unwrap_command, shell=True, check=True)
    fname_phase_unwrapped = glob.glob(os.path.join(path_tmp, 'rawPhase_unwrapped*'))[0]

    return nib.load(fname_phase_unwrapped).get_fdata()
