#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest
import math

from shimmingtoolbox.coils.siemens_basis import siemens_basis

dummy_data = [
    np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)), indexing='ij'),
]


@pytest.mark.parametrize('x,y,z', dummy_data)
def test_normal_siemens_basis(x, y, z):

    basis = siemens_basis(x, y, z)

    # Test for shape
    assert(np.all(basis.shape == (x.shape[0], x.shape[1], x.shape[2], 8)))
    # Test for first value
    assert(math.isclose(basis[0, 0, 0, 0], -4.25760000e-02, rel_tol=1e-09))
