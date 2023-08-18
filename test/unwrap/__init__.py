#!/usr/bin/python3
# -*- coding: utf-8 -*

import nibabel as nib
import numpy as np
import os

from shimmingtoolbox import __dir_testing__


def get_phases_mags():
    """ Load 2 phase echos (as nii) and 2 magnitude echos (as np.array). """

    # Load phase data
    fname_phase1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
    fname_phase2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
    nii_phase_e1 = nib.load(fname_phase1)
    nii_phase_e2 = nib.load(fname_phase2)

    # Scale phase to radians
    phase_e1 = np.interp(nii_phase_e1.get_fdata(), [0, 4096], [-np.pi, np.pi])
    phase_e2 = np.interp(nii_phase_e2.get_fdata(), [0, 4096], [-np.pi, np.pi])

    # Load mag data
    fname_mag1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    fname_mag2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude2.nii.gz')
    nii_mag_e1 = nib.load(fname_mag1)
    nii_mag_e2 = nib.load(fname_mag2)

    # Make tests faster by having the last dim only be 1
    nii_phase_e1 = nib.Nifti1Image(phase_e1[..., :2], nii_phase_e1.affine, header=nii_phase_e1.header)
    nii_phase_e2 = nib.Nifti1Image(phase_e2[..., :2], nii_phase_e1.affine, header=nii_phase_e1.header)
    mag_e1 = nii_mag_e1.get_fdata()[..., :2]
    mag_e2 = nii_mag_e2.get_fdata()[..., :2]

    return nii_phase_e1, nii_phase_e2, mag_e1, mag_e2
