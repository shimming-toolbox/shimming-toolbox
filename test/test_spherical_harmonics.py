#!/usr/bin/python3
# -*- coding: utf-8 -*


import numpy as np

from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics


def test_normal_use():
    [x, y, z] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij')
    orders = np.array(range(0, 2, 1))
    basis = spherical_harmonics(orders, x, y, z)

    assert(basis.ndim == 4)
    assert(basis[:, :, :, 0].shape == x.shape)


def test_wrong_input_dimension():
    [x, y, z] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij')
    orders = np.array(range(0, 2, 1))

    # Call spherical harmonics with wrong dimension
    try:
        spherical_harmonics(orders, x[:, :, 0], y, z)
    except RuntimeError:
        # If an exception occurs, this is the desired behaviour
        return 0

    # If there isn't an error, then there is a problem
    print('\nWrong dimensions for input x does not throw an error')
    assert False


def test_negative_order():
    [x, y, z] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij')
    orders = np.array(range(-1, 2, 1))

    # Call spherical harmonics with negative order
    try:
        spherical_harmonics(orders, x, y, z)
    except RuntimeError:
        # If an exception occurs, this is the desired behaviour
        return 0

    # If there isn't an error, then there is a problem
    print('\nNegative order does not throw an error')
    assert False
