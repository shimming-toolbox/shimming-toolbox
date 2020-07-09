#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Description

"""


import numpy as np
import os
import nibabel as nib


def prelude(complexArray, affine, mask=np.array([-1]), path2UnwrappedPhase="./unwrappedPhase.nii", isUnwrappingIn2D=True, isSavingNiftis=False):
    """

    Args:
        complexArray:
        affine:
        mask:
        path2UnwrappedPhase:
        isUnwrappingIn2D:
        isSavingNiftis:

    Returns:

    """
    # Get absolute path
    absPath = os.path.abspath(path2UnwrappedPhase)
    dataSaveDirectory = os.path.dirname(absPath)

    # Save complex image
    # TODO: Save voxelSize when saving nifti (Maybe affine does it) (if it does, might be easier to just pass entire nibabel object)
    phaseNii = nib.Nifti1Image(np.angle(complexArray[:-1, :-1, 0, 0]), affine)
    magNii = nib.Nifti1Image(np.abs(complexArray[:-1, :-1, 0, 0]), affine)
    nib.save(phaseNii, os.path.join(dataSaveDirectory, "rawPhase.nii"))
    nib.save(magNii, os.path.join(dataSaveDirectory, "mag.nii"))

    # Fill options
    options = " "

    if isUnwrappingIn2D:
        options = options + "-s "

    # Create mask with all ones if mask is not provided
    if not mask.all() == -1:
        assert mask.shape == complexArray.shape, "Mask must be the same shape as the array"
        maskNii = nib.Nifti1Image(mask, affine)

        options = options + "-m "
        options = options + os.path.join(dataSaveDirectory, "mask.nii")
        nib.save(maskNii, os.path.join(dataSaveDirectory, "mask.nii"))

    # Unwrap
    unwrapCommand = "prelude -p " + os.path.join(dataSaveDirectory, "rawPhase") + " -a " + \
                    os.path.join(dataSaveDirectory, "mag") + " -o " + path2UnwrappedPhase + options

    print(unwrapCommand)
    os.system(unwrapCommand)

    # Uncompress
    os.system("gunzip " + path2UnwrappedPhase + ".gz -df")

    unwrappedPhase = nib.load(path2UnwrappedPhase)
    unwrappedPhase = np.array(unwrappedPhase.dataobj)
    # Delete temporary files according to options
    if not isSavingNiftis:
        os.remove(os.path.join(dataSaveDirectory, "mag.nii"))
        os.remove(os.path.join(dataSaveDirectory, "rawPhase.nii"))
        os.remove(os.path.join(dataSaveDirectory, "unwrappedPhase.nii"))
        if not mask.all() == -1:
            os.remove(os.path.join(dataSaveDirectory, "mask.nii"))

    return unwrappedPhase

