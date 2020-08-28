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

    # Test for a value, arbitrarily chose basis[0, 0, 0, 0].
    # The full matrix could be checked to be more thorough but would require explicitly defining the matrix which is
    # 2x2x2x8. -4.25760000e-02 was worked out to be the value that should be in basis[0, 0, 0, 0].
    assert(math.isclose(basis[0, 0, 0, 0], -4.25760000e-02, rel_tol=1e-09))
