#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Description

"""

import numpy as np
import os
import nibabel as nib


def prelude(complex_array, affine, mask=np.array([-1]), path_2_unwrapped_phase="./unwrapped_phase.nii",
            is_unwrapping_in_2d=True, is_saving_nii=False):
    # Get absolute path
    abs_path = os.path.abspath(path_2_unwrapped_phase)
    data_save_directory = os.path.dirname(abs_path)

    # Save complex image
    # TODO: Save voxelSize when saving nifti (Maybe affine does it) (if it does, might be easier to just pass entire nibabel object)
    phase_nii = nib.Nifti1Image(np.angle(complex_array), affine)
    mag_nii = nib.Nifti1Image(np.abs(complex_array), affine)
    nib.save(phase_nii, os.path.join(data_save_directory, "rawPhase.nii"))
    nib.save(mag_nii, os.path.join(data_save_directory, "mag.nii"))

    # Fill options
    options = " "

    if is_unwrapping_in_2d:
        options = options + "-s "

    # Create mask with all ones if mask is not provided
    if not np.any(mask == -1):
        assert mask.shape == complex_array.shape, "Mask must be the same shape as the array"
        maskNii = nib.Nifti1Image(mask, affine)

        options = options + "-m "
        options = options + os.path.join(data_save_directory, "mask.nii")
        nib.save(maskNii, os.path.join(data_save_directory, "mask.nii"))

    # Unwrap
    unwrap_command = "prelude -p " + os.path.join(data_save_directory, "rawPhase") + " -a " + \
                     os.path.join(data_save_directory, "mag") + " -o " + path_2_unwrapped_phase + options

    print(unwrap_command)
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
