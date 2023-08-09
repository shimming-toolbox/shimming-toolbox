#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import json

from shimmingtoolbox.coils.coil import Coil, ScannerCoil
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox import __dir_config_scanner_constraints__


def test_coil_siemens_basis():

    grid_x, grid_y, grid_z = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                         indexing='ij')
    profiles = siemens_basis(grid_x, grid_y, grid_z)

    constraints = {
        "name": "Siemens Basis",
        "coef_sum_max": 40,
        "coef_channel_minmax": [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)],
    }

    a_coil = Coil(profiles, np.eye(4), constraints)

    assert np.array_equal(a_coil.profile, profiles)
    assert a_coil.coef_channel_minmax == constraints['coef_channel_minmax']
    assert a_coil.coef_sum_max == constraints['coef_sum_max']


def test_coil_custom_coil():
    pass
    # Define a custom coil in testing_data


def test_create_scanner_coil_order0():
    sph_contraints = json.load(open(__dir_config_scanner_constraints__))

    scanner_coil = ScannerCoil('ras', (4, 5, 6), np.eye(4), sph_contraints, 0)

    assert scanner_coil.profile[0, 0, 0, 0] == -1.0


def test_create_scanner_coil_order1():
    sph_contraints = json.load(open(__dir_config_scanner_constraints__))

    scanner_coil = ScannerCoil('ras', (4, 5, 6), np.eye(4), sph_contraints, 1)

    assert scanner_coil.profile[0, 0, 0, 0] == -1.0


def test_create_scanner_coil_order2():
    sph_contraints = json.load(open(__dir_config_scanner_constraints__))
    scanner_coil = ScannerCoil('ras', (4, 5, 6), np.eye(4), sph_contraints, 2)

    assert scanner_coil.profile[0, 0, 0, 0] == -1.0
