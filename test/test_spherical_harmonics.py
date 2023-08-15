#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest

from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics

dummy_data = [
    np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij'),
]


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_normal_use(x, y, z):
    orders = np.array(range(0, 2, 1))
    basis = spherical_harmonics(orders, x, y, z)

    assert(basis.ndim == 4)
    assert(basis[:, :, :, 0].shape == x.shape)


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_wrong_input_dimension(x, y, z):
    orders = np.array(range(0, 2, 1))

    with pytest.raises(RuntimeError, match="Input arrays X, Y, and Z must be identically sized"):
        spherical_harmonics(orders, x[:, :, 0], y, z)


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_negative_order(x, y, z):
    orders = np.array(range(-1, 2, 1))

    with pytest.raises(RuntimeError, match="Orders must be positive"):
        spherical_harmonics(orders, x, y, z)
