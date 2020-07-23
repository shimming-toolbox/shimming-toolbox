#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Wrapper to FSL Prelude (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#PRELUDE_.28phase_unwrapping.29)
"""

import numpy as np
import os
import nibabel as nib
import subprocess


def prelude(wrapped_phase, mag, affine, mask=np.array([-1]), path_2_unwrapped_phase='./unwrapped_phase.nii',
            is_unwrapping_in_2d=True, is_saving_nii=False):
    """wrapper to FSL prelude

    This function enables phase unwrapping by calling FSL prelude on the command line. A mask can be provided to mask
    the complex image provided. 2D unwrapping can be turned off. The output path can be specified. THe temporary niis
    can optionally be saved.

    Args:
        wrapped_phase (numpy.ndarray): 3D radian numpy array to perform phase unwrapping
        mag (numpy.ndarray): 3D magnitude numpy array corresponding to the phase array
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be acquired by :
            nii = nib.load("nii_path")
            affine = nii.affine
        mask (numpy.ndarray, optional): numpy array of booleans with shape of `complex_array` to mask during phase unwrapping
        path_2_unwrapped_phase (string, optional): relative or absolute path to output the nii unwrapped phase
        is_unwrapping_in_2d (bool, optional): prelude parameter to unwrap un 2d
        is_saving_nii (bool, optional): specify whether `complex_array`, `affine`, `mask` and `unwrapped_phase` nii files will be
        saved

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude
    """
    # Get absolute path
    abs_path = os.path.abspath(path_2_unwrapped_phase)
    data_save_directory = os.path.dirname(abs_path)

    # Make sure directory exists, if not create it
    if not os.path.exists(data_save_directory):
        print('\nCreating directory for unwrapped phase at {}'.format(data_save_directory))
        os.mkdir(data_save_directory)

    # Make sure phase and mag are the right shape
    if wrapped_phase.ndim != 3:
        raise RuntimeError('wrapped_phase must be 3d')
    if wrapped_phase.shape != mag.shape:
        raise RuntimeError('The magnitude image (mag) must be the same shape as wrapped_phase')

    # Save phase and mag images
    phase_nii = nib.Nifti1Image(wrapped_phase, affine)
    mag_nii = nib.Nifti1Image(mag, affine)
    nib.save(phase_nii, os.path.join(data_save_directory, 'rawPhase.nii'))
    nib.save(mag_nii, os.path.join(data_save_directory, 'mag.nii'))

    # Fill options
    options = ' '
    if is_unwrapping_in_2d:
        options = '-s '

    # Add mask data and options if there is a mask provided
    if not np.any(mask == -1):
        # TODO: Make sure values are either 1 or 0
        if mask.shape != wrapped_phase.shape:
            raise RuntimeError('Mask must be the same shape as wrapped_phase')
        mask_nii = nib.Nifti1Image(mask, affine)

        options += '-m '
        options += os.path.join(data_save_directory, 'mask.nii')
        nib.save(mask_nii, os.path.join(data_save_directory, 'mask.nii'))

    # Unwrap
    unwrap_command = 'prelude -p {} -a {} -o {} {}'.format(os.path.join(data_save_directory, 'rawPhase'),
                                                           os.path.join(data_save_directory, 'mag'),
                                                           path_2_unwrapped_phase, options)
    subprocess.run(unwrap_command, shell=True, check=True)

    # Uncompress
    subprocess.run(['gunzip', path_2_unwrapped_phase + '.gz', '-df'], check=True)
    unwrapped_phase = nib.load(path_2_unwrapped_phase)
    unwrapped_phase = unwrapped_phase.get_fdata()

    # Delete temporary files according to options
    if not is_saving_nii:
        os.remove(os.path.join(data_save_directory, 'mag.nii'))
        os.remove(os.path.join(data_save_directory, 'rawPhase.nii'))
        os.remove(os.path.join(data_save_directory, 'unwrapped_phase.nii'))
        if not np.any(mask == -1):
            os.remove(os.path.join(data_save_directory, 'mask.nii'))

    return unwrapped_phase
