#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np

from shimmingtoolbox.coils.coordinates import generate_meshgrid


def test_generate_meshgrid():
    """Test to verify generate_meshgrid outputs the correct scanner coordinates from input voxels"""

    affine = np.array([[0., 0.,    3., -3.61445999],
                      [-2.91667008, 0., 0., 101.76699829],
                      [0., 2.91667008, 0., -129.85464478],
                      [0., 0., 0., 1.]])

    nx = 2
    ny = 2
    nz = 2
    coord = generate_meshgrid((nx, ny, nz), affine)

    expected = [np.array([[[-3.61445999, -0.61445999],
                           [-3.61445999, -0.61445999]],
                          [[-3.61445999, -0.61445999],
                           [-3.61445999, -0.61445999]]]),
                np.array([[[101.76699829, 101.76699829],
                           [101.76699829, 101.76699829]],
                          [[98.85032821,  98.85032821],
                           [98.85032821,  98.85032821]]]),
                np.array([[[-129.85464478, -129.85464478],
                           [-126.9379747, -126.9379747]],
                          [[-129.85464478, -129.85464478],
                           [-126.9379747, -126.9379747]]])]

    assert(np.all(np.isclose(coord, expected)))
