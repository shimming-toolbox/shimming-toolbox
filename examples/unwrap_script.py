#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Description

"""


from shimmingtoolbox.unwrap import unwrap_phase
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cmath


def main():

    # Import phase
    phasePath = ""
    phaseNii = nib.load(phasePath)
    phase = np.array(phaseNii.dataobj)

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
