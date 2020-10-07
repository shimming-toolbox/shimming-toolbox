#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import nibabel
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_shimmingtoolbox__


def realtime_zshim():
    fname = os.path.join(__dir_shimmingtoolbox__, __dir_testing__, 'nifti', 'sub-example', 'fmap', 'sub-example_fieldmap.nii.gz')
    nii = nibabel.load(fname)
    fieldmaps = nii.get_fdata()
    affine = nii.affine

    nx, ny, nz, nt = fieldmaps.shape

    # Set up coils
    coord_vox = np.meshgrid(np.array(range(nx)), np.array(range(ny)), np.array(range(nz)), indexing='ij')
    coord_phys = [np.zeros_like(coord_vox[0]), np.zeros_like(coord_vox[1]), np.zeros_like(coord_vox[2])]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                coord_phys_list = np.dot([coord_vox[i][ix, iy, iz] for i in range(3)], affine[0:3, 0:3]) + affine[0:3,
                                                                                                           3]
                for i in range(3):
                    coord_phys[i][ix, iy, iz] = coord_phys_list[i]

    # coord_phys was checked and has the correct scanner coordinates
    # TODO: Better code ^ and add as a function

    basis = siemens_basis(coord_phys[0], coord_phys[1], coord_phys[2])

    # Set up mask
    full_mask = shapes(fieldmaps[:, :, :, 0], 'cube', center_dim1=round(nx/2)-5, len_dim1=40, len_dim2=40, len_dim3=nz)

    currents = np.zeros([8, nt])
    shimmed = np.zeros_like(fieldmaps)
    masked_fieldmaps = np.zeros_like(fieldmaps)
    masked_shimmed = np.zeros_like(shimmed)
    for i_t in range(nt):
        currents[:, i_t] = sequential_zslice(fieldmaps[:, :, :, i_t], basis, full_mask, z_slices=np.array(range(nz)))
        shimmed[:, :, :, i_t] = fieldmaps[:, :, :, i_t] + np.sum(currents[:, i_t] * basis, axis=3, keepdims=False)
        masked_fieldmaps[:, :, :, i_t] = full_mask * fieldmaps[:, :, :, i_t]
        masked_shimmed[:, :, :, i_t] = full_mask * shimmed[:, :, :, i_t]

    i_t = 0
    # Plot results
    fig = Figure(figsize=(10, 10))
    # FigureCanvas(fig)
    ax = fig.add_subplot(2, 2, 1)
    im = ax.imshow(masked_fieldmaps[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Masked unshimmed")
    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(masked_shimmed[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Masked shimmed")
    ax = fig.add_subplot(2, 2, 3)
    im = ax.imshow(fieldmaps[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Unshimmed")
    ax = fig.add_subplot(2, 2, 4)
    im = ax.imshow(shimmed[:-1, :-1, 0, i_t])
    fig.colorbar(im)
    ax.set_title("Shimmed")

    print(f"\nThe associated current coefficients are : {currents[:, i_t]}")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'realtime_zshim_plot.png')
    fig.savefig(fname_figure)
    return fname_figure


if __name__ == '__main__':
    realtime_zshim()
