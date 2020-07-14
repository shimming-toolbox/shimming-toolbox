#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""prelude is a wrapper function to FSL Prelude


"""

import numpy as np
import os
import nibabel as nib


def prelude(complex_array, affine, mask=np.array([-1]), path_2_unwrapped_phase="./unwrapped_phase.nii",
            is_unwrapping_in_2d=True, is_saving_nii=False):
    """wrapper to FSL prelude

    This function enables phase unwrapping by calling FSL prelude on the command line. A mask can be provided to mask
    the complex image provided. 2D unwrapping can be turned off. The output path can be specified. THe temporary niis
    can optionally be saved.

    Args:
        complex_array (numpy.ndarray): 3D complex values numpy array to perform phase unwrapping
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be acquired by :
            nii = nib.load("nii_path")
            affine = nii.affine
        mask (numpy.ndarray): numpy array of booleans with shape of `complex_array` to mask during phase unwrapping
        path_2_unwrapped_phase (string): relative or absolute path to output the nii unwrapped phase
        is_unwrapping_in_2d (bool): prelude parameter to unwrap un 2d
        is_saving_nii (bool): specify whether `complex_array`, `affine`, `mask` and `unwrapped_phase` nii files will be
        saved

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude
    """
    # Get absolute path
    abs_path = os.path.abspath(path_2_unwrapped_phase)
    data_save_directory = os.path.dirname(abs_path)

    # Save complex image TODO: Save voxelSize when saving nifti (Maybe affine does it) (if it does, might be easier
    #  to just pass entire nibabel object)
    phase_nii = nib.Nifti1Image(np.angle(complex_array), affine)
    mag_nii = nib.Nifti1Image(np.abs(complex_array), affine)
    nib.save(phase_nii, os.path.join(data_save_directory, "rawPhase.nii"))
    nib.save(mag_nii, os.path.join(data_save_directory, "mag.nii"))

    # Fill options
    options = " "

    if is_unwrapping_in_2d:
        options += "-s "

    # Add mask data and options if there is a mask provided
    if not np.any(mask == -1):
        assert mask.shape == complex_array.shape, "Mask must be the same shape as the array"
        mask_nii = nib.Nifti1Image(mask, affine)

        options += "-m "
        options += os.path.join(data_save_directory, "mask.nii")
        nib.save(mask_nii, os.path.join(data_save_directory, "mask.nii"))

    # Unwrap
    unwrap_command = "prelude -p {} -a {} -o {} {}".format(os.path.join(data_save_directory, "rawPhase"),
                                                           os.path.join(data_save_directory, "mag"),
                                                           path_2_unwrapped_phase, options)
    os.system(unwrap_command)

    # Uncompress
    os.system("gunzip " + path_2_unwrapped_phase + ".gz -df")
    unwrapped_phase = nib.load(path_2_unwrapped_phase)
    unwrapped_phase = unwrapped_phase.get_fdata()

    # Delete temporary files according to options
    if not is_saving_nii:
        os.remove(os.path.join(data_save_directory, "mag.nii"))
        os.remove(os.path.join(data_save_directory, "rawPhase.nii"))
        os.remove(os.path.join(data_save_directory, "unwrapped_phase.nii"))
        if not np.any(mask == -1):
            os.remove(os.path.join(data_save_directory, "mask.nii"))

    return unwrapped_phase
