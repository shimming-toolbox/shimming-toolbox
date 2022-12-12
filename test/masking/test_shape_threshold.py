# coding: utf-8

import numpy as np
import pytest

from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.masking.shapes import shapes


dummy_data = [
    (np.array([3, 4]), np.array([False,  True])),
    (np.array([[3, 4], [5, 6]]), np.array([[False,  True], [True, True]])),
]

dummy_data_shape_square = [
    (np.ones([3, 2]), 'square', np.array([[True, True], [False, False], [False, False]])),
]

dummy_data_shape_cube = [
    (np.ones([4, 3, 2]),
     'cube',
     np.array([[[False, False], [False, False], [False, False]],
               [[False, True], [False, True], [False, True]],
               [[False, False], [False, False], [False, False]],
               [[False, False], [False, False], [False, False]]])),
]

dummy_data_shape_sphere = [
    (np.ones([4, 3, 2]),
     'sphere',
     np.array([[[True, True], [True, True], [True, True]],
               [[True, True], [True, True], [True, True]],
               [[True, True], [True, True], [True, True]],
               [[False, False], [False, True], [False, False]]])),
]


@pytest.mark.parametrize('data,expected', dummy_data)
def test_threshold(data, expected):
    assert np.all(threshold(data, thr=3) == expected)


@pytest.mark.parametrize('data,shape,expected', dummy_data_shape_square)
def test_mask_square(data, shape, expected):
    assert(np.all(shapes(data, shape, center_dim1=0, center_dim2=1, len_dim1=1, len_dim2=3) == expected))


def test_mask_square_wrong_dims():
    data = np.ones([2, 2, 2])
    with pytest.raises(ValueError, match="shape_square only allows for 2 dimensions"):
        shapes(data, 'square', center_dim1=0, center_dim2=1, len_dim1=1, len_dim2=3)


@pytest.mark.parametrize('data,shape,expected', dummy_data_shape_cube)
def test_mask_cube(data, shape, expected):
    assert(np.all(shapes(data, shape, center_dim1=1, center_dim2=1, center_dim3=1, len_dim1=1, len_dim2=3, len_dim3=1)
                  == expected))


def test_mask_cube_wrong_dims():
    data = np.ones([2, 2])
    with pytest.raises(ValueError, match="shape_cube only allows for 2 dimensions"):
        shapes(data, 'cube', center_dim1=1, center_dim2=1, center_dim3=1, len_dim1=1, len_dim2=3, len_dim3=1)


@pytest.mark.parametrize('data,shape,expected', dummy_data_shape_sphere)
def test_mask_sphere(data, shape, expected):
    assert(np.all(shapes(data, shape, radius=2, center_dim1=1, center_dim2=1, center_dim3=1) == expected))


def test_mask_sphere_wrong_dims():
    data = np.ones([2, 2])
    with pytest.raises(ValueError, match="shape_sphere only allows for 3 dimensions"):
        shapes(data, 'sphere', radius=1)
