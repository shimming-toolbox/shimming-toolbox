#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest

from shimmingtoolbox.coils.biot_savart import biot_savart

dummy_data = [
    ([(0, 0, -0.5)], [(0, 0, 1)], [1, ], [250, ], (0, 0, 0), (0, 0, 1), (1, 1, 3), (0.4495881427866065e-3,
                                                                                    2.2214414690791835e-4,
                                                                                    1.0723951147113824e-4)),
    ([(-0.5, 0, 0)], [(1, 0, 0)], [1, ], [250, ], (0, 0, 0), (1, 0, 0), (3, 1, 1), (0, 0, 0)),
    ([(0, -0.5, 0)], [(0, 1, 0)], [1, ], [250, ], (0, 0, 0), (0, 1, 0), (1, 3, 1), (0, 0, 0)),
]


@pytest.mark.parametrize('centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n, axis_answers', dummy_data)
def test_normal_use(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n, axis_answers):
    basis = biot_savart(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n)

    assert(basis.ndim == 4)
    assert(basis[:, :, :, 0].shape == fov_n)

    # Check expected values
    actual = np.round(basis[:, :, :, 0].reshape(len(axis_answers)), decimals=7)
    expected = np.round(np.asarray(axis_answers), decimals=7)
    assert(np.all(actual == expected))
