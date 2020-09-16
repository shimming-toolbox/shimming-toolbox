#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from matplotlib.figure import Figure

from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes


def test_zslice():

    # Set up unshimmed fieldmap
    num_vox = 100
    model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
    model_obj.generate_deltaB0('linear', [0.05, 0.01])
    tr = 0.025  # in s
    te = [0.004, 0.008]  # in s
    model_obj.simulate_measurement(tr, te)
    phase_meas1 = model_obj.get_phase()
    phase_e1 = phase_meas1[:, :, 0, 0]
    phase_e2 = phase_meas1[:, :, 0, 1]
    b0_map = (phase_e2 - phase_e1)/(te[1] - te[0])
    nz = 3

    # Construct synthetic field map based on a manipulation of model_obj across slices
    unshimmed = np.zeros([num_vox, num_vox, nz])
    unshimmed[:, :, 0] = b0_map
    unshimmed[:, :, 1] = (np.rot90(unshimmed[:, :, 0]) + unshimmed[:, :, 0]) / 2
    unshimmed[:, :, 2] = unshimmed[:, :, 0] ** 2

    # Set up coil profile
    x, y, z = np.meshgrid(
        np.array(range(int(-num_vox/2), int(num_vox/2))),
        np.array(range(int(-num_vox/2), int(num_vox/2))),
        np.array(range(nz)),
        indexing='ij')
    coils = siemens_basis(x, y, z)

    # Set up mask
    full_mask = shapes(unshimmed, 'cube', len_dim1=40, len_dim2=40, len_dim3=nz)

    # Optimize
    currents = sequential_zslice(unshimmed, coils, full_mask, np.array(range(nz)))

    # Calculate theoretical shimmed map
    shimmed = unshimmed + np.sum(currents * coils, axis=3, keepdims=False)

    # Plot results
    fig = Figure(figsize=(10, 10))
    # FigureCanvas(fig)
    for z in range(3):
        ax = fig.add_subplot(3, 2, 2 * z + 1)
        im = ax.imshow(full_mask[:, :, z] * unshimmed[:, :, z])
        fig.colorbar(im)
        ax.set_title(f"Unshimmed {z}")
        ax = fig.add_subplot(3, 2, 2 * z + 2)
        im = ax.imshow(full_mask[:, :, z] * shimmed[:, :, z])
        fig.colorbar(im)
        ax.set_title(f"Shimmed {z}")

    fname_figure = os.path.join(os.path.curdir, 'shimmed_plot.png')
    fig.savefig(fname_figure)

    # TODO: assert
