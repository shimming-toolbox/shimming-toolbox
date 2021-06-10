#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.siemens_basis import siemens_basis


def test_coil_siemens_basis():

    grid_x, grid_y, grid_z = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                         indexing='ij')
    profiles = siemens_basis(grid_x, grid_y, grid_z)

    constraints = {
        "coef_sum_max": 40,
        "coef_channel_minmax": [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]
    }

    a_coil = Coil(profiles, np.eye(4), constraints)

    assert np.array_equal(a_coil.profile, profiles)
    assert a_coil.coef_channel_minmax == constraints['coef_channel_minmax']
    assert a_coil.coef_sum_max == constraints['coef_sum_max']


def test_coil_custom_coil():
    pass
    # Define a custom coil in testing_data
