#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from shimmingtoolbox.optimizer.sequential import sequential_zslice
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.simulate.numerical_model import NumericalModel
from shimmingtoolbox.masking.shapes import shapes


def test_zslice():

    # Set up unshimmed fieldmap
    num_vox = 100
    model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
    model_obj.generate_deltaB0('linear', [0.05, 0.01])
    TR = 0.025
    TE = [0.004, 0.008]
    model_obj.simulate_measurement(TR, TE)
    phase_meas1 = model_obj.get_phase()
    phase_e1 = phase_meas1[:, :, 1, 0]
    phase_e2 = phase_meas1[:, :, 1, 1]
    B0_map = (phase_e2 - phase_e1)/(TE[1] - TE[0])
    unshimmed = np.zeros([num_vox, num_vox, 3])

    # Probably a better way
    for x in unshimmed.shape[2]:
        unshimmed[:, :, x] = B0_map

    # Set up coil profile
    x, y, z = np.meshgrid(np.array(range(int(num_vox/2), int(num_vox/2 + 1))), np.array(range(int(-num_vox/2), int(num_vox/2 + 1))), np.array(range(-1, 2)), indexing='ij')
    coils = siemens_basis(x, y, z)

    # Set up mask
    full_mask = shapes(unshimmed, 'cube', len_dim1=20, len_dim2=20, len_dim3=3)


    currents = sequential_zslice(unshimmed, coils, full_mask, z_slices)
