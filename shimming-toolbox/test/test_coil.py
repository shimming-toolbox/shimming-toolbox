#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy

import numpy as np
import json

from shimmingtoolbox.coils.coil import Coil, ScannerCoil, get_scanner_constraints, SCANNER_CONSTRAINTS
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox import __config_scanner_constraints__

shim_settings = {
    '0': [1],
    '1': [1, 1, 1],
    '2': [1, 1, 1, 1, 1],
    '3': [1, 1, 1, 1]
}


def test_coil_siemens_basis():
    grid_x, grid_y, grid_z = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                         indexing='ij')
    profiles = siemens_basis(grid_x, grid_y, grid_z)

    constraints = {
        "name": "Siemens Basis",
        "coef_sum_max": 40,
        "coef_channel_minmax": {"coil": [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]},
    }

    a_coil = Coil(profiles, np.eye(4), constraints)

    assert np.array_equal(a_coil.profile, profiles)
    assert a_coil.coef_channel_minmax == constraints['coef_channel_minmax']
    assert a_coil.coef_sum_max == constraints['coef_sum_max']


def test_custom_coil_coefs_used():
    grid_x, grid_y, grid_z = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                         indexing='ij')
    profiles = siemens_basis(grid_x, grid_y, grid_z)

    constraints = {
        "name": "Siemens Basis",
        "coef_sum_max": 40,
        "coef_channel_minmax": {"coil": [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]},
        "coefs_used": {"coil": [-2, -1, 0, 1, 2, 2, 2, 2]},
    }

    a_coil = Coil(profiles, np.eye(4), constraints)

    assert np.array_equal(a_coil.profile, profiles)
    assert a_coil.coef_channel_minmax == {
        "coil": [[0, 4], [-1, 3], [-2, 2], [-3, 1], [-4, 0], [-4, 0], [-4, 0], [-4, 0]]}
    assert a_coil.coef_sum_max == constraints['coef_sum_max']


def test_scanner_coil_coefs_used():
    grid_x, grid_y, grid_z = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                         indexing='ij')
    profiles = siemens_basis(grid_x, grid_y, grid_z, (0, 1, 2, 3))

    constraints = {
        "name": "Siemens Basis",
        "coef_sum_max": 40,
        "coef_channel_minmax": {"0": [[-2, 2]],
                                "1": [[-2, 2], [-2, 2], [-2, 2]],
                                "2": [[-2, 2], [None, 2], [-2, None], [None, None], [-2, 2]],
                                "3": [[None, None], [None, None], [None, None], [None, None]]},
        "coefs_used": {"0": [-2], "1": [-1, 0, 1], "2": [2, 2, 2, 2, None], "3": None},
    }

    a_coil = Coil(profiles, np.eye(4), constraints)

    assert np.array_equal(a_coil.profile, profiles)
    assert a_coil.coef_channel_minmax == {"0": [[0, 4]],
                                          "1": [[-1, 3], [-2, 2], [-3, 1]],
                                          "2": [[-4, 0], [-np.inf, 0], [-4, np.inf], [-np.inf, np.inf], [-2, 2]],
                                          "3": [[-np.inf, np.inf],
                                                [-np.inf, np.inf],
                                                [-np.inf, np.inf],
                                                [-np.inf, np.inf]]}
    assert a_coil.coef_sum_max == constraints['coef_sum_max']


def test_create_scanner_coil_order0():
    sph_contraints = json.load(open(__config_scanner_constraints__))

    scanner_coil = ScannerCoil((4, 5, 6), np.eye(4), sph_contraints, [0])

    assert scanner_coil.profile[0, 0, 0, 0] == -1.0


def test_create_scanner_coil_order1():
    sph_contraints = json.load(open(__config_scanner_constraints__))

    scanner_coil = ScannerCoil((4, 5, 6), np.eye(4), sph_contraints, [0, 1])

    assert scanner_coil.profile[0, 0, 0, 0] == -1.0


def test_create_scanner_coil_siemens_order2():
    sph_contraints = json.load(open(__config_scanner_constraints__))
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
    scanner_coil = ScannerCoil((3, 3, 3), affine, sph_contraints, [0, 1, 2], 'SIEMENS')

    assert np.allclose(scanner_coil.profile[:, 1, :, 5], np.array([[8.5154957e-05, 0.0000000e+00, -8.5154957e-05],
                                                                   [-0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                                                                   [-8.5154957e-05, -0.0000000e+00, 8.5154957e-05]]))


def test_create_scanner_coil_philips():
    sph_contraints = json.load(open(__config_scanner_constraints__))
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
    scanner_coil = ScannerCoil((3, 3, 3), affine, sph_contraints, [0, 1, 2], 'PHILIPS')

    assert np.allclose(scanner_coil.profile[1, :, :, 5], np.array([[-8.5154957e-02 / 2, 0, 8.5154957e-02 / 2],
                                                                   [0, 0, 0],
                                                                   [8.5154957e-02 / 2, 0, -8.5154957e-02 / 2]]))


def test_create_scanner_coil_ge():
    sph_contraints = json.load(open(__config_scanner_constraints__))
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
    scanner_coil = ScannerCoil((3, 3, 3), affine, sph_contraints, [0, 1, 2], 'GE')

    assert np.allclose(scanner_coil.profile[1, :, :, 5], np.array([[-5.6703165e-05, -5.0049942e-05, -4.3501609e-05],
                                                                   [-5.9169445e-05, -5.2430000e-05, -4.5795445e-05],
                                                                   [-6.1538609e-05, -5.4712942e-05, -4.7992165e-05]]))


def test_get_scanner_constraints():
    orders = [0, 1, 2, 3]
    for manufacturer in SCANNER_CONSTRAINTS.keys():
        for model in SCANNER_CONSTRAINTS[manufacturer].keys():
            constraints = get_scanner_constraints(model, orders, manufacturer, shim_settings)
            for order in orders:
                if SCANNER_CONSTRAINTS[manufacturer][model][str(order)]:
                    assert np.all(np.isclose(constraints['coef_channel_minmax'][str(order)],
                                             SCANNER_CONSTRAINTS[manufacturer][model][str(order)]))


def test_get_scanner_constraints_specific_orders():
    orders = [0, 2]
    constraints = get_scanner_constraints("Prisma_fit", orders, "Siemens", shim_settings)
    for order in orders:
        assert np.all(np.isclose(constraints['coef_channel_minmax'][str(order)],
                                 SCANNER_CONSTRAINTS["Siemens"]["Prisma_fit"][str(order)]))
    assert not constraints['coef_channel_minmax']["1"]


def test_get_scanner_constraints_external():
    orders = [0, 1, 2, 3]
    constraints_external = {
        'coefs_used': copy.deepcopy(shim_settings),
        'coef_channel_minmax': copy.deepcopy(SCANNER_CONSTRAINTS["Siemens"]["Prisma_fit"])
    }

    constraints_external['coef_channel_minmax']['0'] = [[-1000, 1000]]
    constraints_external['coef_channel_minmax']['1'] = None
    constraints_external['coef_channel_minmax']['3'] = [[-3000, 3000], [-3000, 3000], [-3000, 3000], [-3000, 3000]]
    constraints_external['coefs_used']['0'] = [2]
    constraints_external['coefs_used']['1'] = None

    tmp_shim_settings = copy.deepcopy(shim_settings)
    tmp_shim_settings['3'] = None

    constraints = get_scanner_constraints("Prisma_fit", orders, "Siemens", tmp_shim_settings, constraints_external)

    # 0 - External constraints are known, internal constraints are known
    assert np.all(np.isclose(constraints['coef_channel_minmax']['0'], constraints_external['coef_channel_minmax']['0']))
    assert constraints['coefs_used']['0'] == constraints_external['coefs_used']['0']
    # 1 - External constraints are not known, internal constraints are known
    assert np.all(np.isclose(constraints['coef_channel_minmax']['1'], SCANNER_CONSTRAINTS["Siemens"]["Prisma_fit"]['1']))
    assert constraints['coefs_used']['1'] == tmp_shim_settings['1']
    # 3 - External constraints are known, internal constraints are not known
    assert np.all(np.isclose(constraints['coef_channel_minmax']['3'], constraints_external['coef_channel_minmax']['3']))
    assert constraints['coefs_used']['3'] == constraints_external['coefs_used']['3']

    # 3 - External constraints are not known, internal constraints are not known
    constraints_external['coef_channel_minmax']['3'] = []
    constraints_external['coefs_used']['3'] = None
    tmp_shim_settings['3'] = None
    constraints = get_scanner_constraints("Prisma_fit", orders, "Siemens", tmp_shim_settings, constraints_external)

    assert constraints['coef_channel_minmax']['3'] == [[None, None], [None, None], [None, None], [None, None]]
    assert constraints['coefs_used']['3'] == [0, 0, 0, 0]
