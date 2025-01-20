#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import json

from shimmingtoolbox.coils.coil import Coil, ScannerCoil, get_scanner_constraints, SCANNER_CONSTRAINTS
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox import __config_scanner_constraints__


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


def test_coil_custom_coil():
    pass
    # Define a custom coil in testing_data


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


def test_create_scanner_coil_isocenter():
    sph_contraints = json.load(open(__config_scanner_constraints__))
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
    scanner_coil = ScannerCoil((3, 3, 3), affine, sph_contraints, [0, 1, 2], 'SIEMENS',
                               isocenter=np.array([0, 0, 0]))

    assert np.all(scanner_coil.isocenter == [0, 0, 0])
    assert np.allclose(scanner_coil.profile[0, 0, 1, :], [-1, 4.25774785e-02, -4.25774785e-02,
                                                          0, -4.25774785e-05, 0,
                                                          0, 0, -8.51549570e-05])
    assert np.allclose(scanner_coil.profile[0, 1, 0, :], [-1, 4.25774785e-02, 0,
                                                          4.25774785e-02, 2.12887393e-05, 8.51549570e-05,
                                                          0, 4.25774785e-05, 0])
    assert np.allclose(scanner_coil.profile[1, 0, 0, :], [-1, 0, -4.25774785e-02,
                                                          4.25774785e-02, 2.12887393e-05, 0,
                                                          -8.51549570e-05, -4.25774785e-05, 0])


def test_create_scanner_coil_not_isocenter():
    sph_contraints = json.load(open(__config_scanner_constraints__))
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
    scanner_coil = ScannerCoil((3, 3, 3), affine, sph_contraints, [0, 1, 2], 'SIEMENS',
                               isocenter=np.array([3, 4, 5]))

    # assert np.all(scanner_coil.isocenter == [3, 4, 5])
    assert np.allclose(scanner_coil.profile[0, 0, 1, :], [-1,  1.70309914e-01, -2.12887393e-01,
                                                          2.12887393e-01, 1.91598653e-04,  1.70309914e-03,
                                                          -2.12887393e-03, -3.83197307e-04, -1.70309914e-03])
    assert np.allclose(scanner_coil.profile[0, 1, 0, :], [-1,  1.70309914e-01, -1.70309914e-01,
                                                          2.55464871e-01, 8.51549570e-04,  2.04371897e-03,
                                                          -2.04371897e-03,  8.34277965e-20, -1.36247931e-03])
    assert np.allclose(scanner_coil.profile[1, 0, 0, :], [-1,  1.27732436e-01, -2.12887393e-01,
                                                          2.55464871e-01, 8.08972092e-04, 1.53278923e-03,
                                                          -2.55464871e-03, -6.81239656e-04, -1.27732436e-03])


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
            constraints = get_scanner_constraints(model, orders, manufacturer)
            for order in orders:
                if SCANNER_CONSTRAINTS[manufacturer][model][str(order)]:
                    assert np.all(np.isclose(constraints['coef_channel_minmax'][str(order)],
                                             SCANNER_CONSTRAINTS[manufacturer][model][str(order)]))


def test_get_scanner_constraints_specific_orders():
    orders = [0, 2]
    constraints = get_scanner_constraints("Prisma_fit", orders, "Siemens")
    for order in orders:
        assert np.all(np.isclose(constraints['coef_channel_minmax'][str(order)],
                                 SCANNER_CONSTRAINTS["Siemens"]["Prisma_fit"][str(order)]))
    assert not constraints['coef_channel_minmax']["1"]
