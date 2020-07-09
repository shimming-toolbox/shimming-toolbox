#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" This script will:
- download 2-echo phase fieldmap data
- do the subtraction
- unwrap phase difference
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import glob

import nibabel as nib

from shimmingtoolbox.unwrap import unwrap_phase


def main():

    # Download example data
    url = 'https://github.com/shimming-toolbox/data-testing/archive/r20200709.zip'
    filename = 'data-testing.zip'
    # TODO: replace with download function
    os.system('curl -o {} -L {}'.format(filename, url))
    with ZipFile(filename, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()
    os.remove(filename)
    # TODO: use systematic name for data-testing (could be in metadata of shimmingtoolbox
    path_data = glob.glob('data-test*')[0]

    # Open phase data
    fname_phases = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
    nii_e1 = nib.load(fname_phases[0])
    nii_e2 = nib.load(fname_phases[1])

    # Subtract phase
    phase_diff = nii_e2.get_fdata() - nii_e1.get_fdata()

    magPath = ""
    magNii = nib.load(magPath)
    mag = np.array(magNii.dataobj)

    # Convert to radians (Assumes there are wraps)
    phase = - np.min(phase) + phase
    phase = phase / 4096 * 2 * np.pi - np.pi

    # print(wrappedPhase)
    # print(nifti.header)
    # print(wrappedPhase.shape)

    # Need to iterate on 4th dimension
    mappingAlgo = "prelude"
    complexArray = mag / 4096 * np.exp(1j * phase)
    affine = phaseNii.affine
    # print(complexArray)

    unwrappedPhase = unwrap_phase(complexArray, affine, mappingAlgo)


    plt.figure(figsize = (10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.angle(complexArray[:-1, :-1, 0, 0]))
    plt.title("Wrapped")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(unwrappedPhase[:-1, :-1])
    plt.title("Unwrapped")
    plt.colorbar()
    # plt.show()
    plt.savefig("myplot.png")


if __name__ == '__main__':
    main()
