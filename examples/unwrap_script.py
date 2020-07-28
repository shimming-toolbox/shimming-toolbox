#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" This script will:
- download 2-echo phase fieldmap data
- unwrap phase difference
- do the subtraction
- save wrapped and unwrapped plot of first X,Y volume as myplot.png in current directory
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess

import nibabel as nib

from shimmingtoolbox.unwrap import prelude


def main():

    # Download example data
    subprocess.run(['st_download_data testing_data'], shell=True, check=True)

    # Open phase data
    path_data = glob.glob(os.path.join('.', 'testing_data*'))[0]
    fname_phases = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
    if len(fname_phases) > 2:
        raise IndexError('Phase data parsing is wrongly parsed')

    nii_phase_e1 = nib.load(fname_phases[0])
    nii_phase_e2 = nib.load(fname_phases[1])

    # Scale to phase to radians
    phase_e1 = nii_phase_e1.get_fdata() / 4096.0 * 2.0 * np.pi - np.pi
    phase_e2 = nii_phase_e2.get_fdata() / 4096.0 * 2.0 * np.pi - np.pi

    # Open mag data
    fname_mags = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*magnitude*.nii.gz'))

    nii_mag_e1 = nib.load(fname_mags[0])
    nii_mag_e2 = nib.load(fname_mags[1])

    # TODO: create mask
    # Call SCT or user defined mask
    # mask = np.ones(phase_e1.shape)

    # Call prelude to unwrap the phase
    unwrapped_phase_e1 = prelude(phase_e1, nii_mag_e1.get_fdata(), nii_phase_e1.affine)
    unwrapped_phase_e2 = prelude(phase_e2, nii_mag_e2.get_fdata(), nii_phase_e2.affine)

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(phase_e1[:-1, :-1, 0])
    plt.title("Wrapped e1")
    plt.subplot(2, 2, 2)
    plt.imshow(phase_e2[:-1, :-1, 0])
    plt.title("Wrapped e2")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(unwrapped_phase_e1[:-1, :-1, 0])
    plt.title("Unwrapped e1")
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(unwrapped_phase_e2[:-1, :-1, 0])
    plt.title("Unwrapped e2")
    plt.colorbar()

    # plt.show()
    plt.savefig("unwrap_phase_plot.png")


if __name__ == '__main__':
    main()
