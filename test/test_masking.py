# coding: utf-8

import numpy as np
import pytest
from matplotlib.figure import Figure

import shimmingtoolbox as shim
import shimmingtoolbox.masking.threshold
import shimmingtoolbox.masking.shape


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


@pytest.mark.parametrize('data,expected', dummy_data)
def test_threshold(data, expected):
    assert np.all(shim.masking.threshold.threshold(data, thr=3) == expected)


@pytest.mark.parametrize('data,shape,expected', dummy_data_shape_square)
def test_mask_square(data, shape, expected):
    assert(np.all(shim.masking.shape.shape(data, shape, center_dim1=0, center_dim2=1, len_dim1=1, len_dim2=3) ==
                  expected))


def test_mask_square_wrong_dims():
    data = np.ones([2, 2, 2])
    try:
        shim.masking.shape.shape(data, 'square', center_dim1=0, center_dim2=1, len_dim1=1, len_dim2=3)
    except RuntimeError:
        # If an exception occurs, this is the desired behaviour since the mask is the wrong dimensions
        return 0

    # If there isn't an error, then there is a problem
    print('\n3D input data does not throw an error')
    assert False


@pytest.mark.parametrize('data,shape,expected', dummy_data_shape_cube)
def test_mask_cube(data, shape, expected):
    assert(np.all(shim.masking.shape.shape(data, shape, center_dim1=1, center_dim2=1, center_dim3=1, len_dim1=1,
                                           len_dim2=3, len_dim3=1) == expected))


def test_mask_cube_wrong_dims():
    data = np.ones([2, 2])
    try:
        shim.masking.shape.shape(data, 'cube', center_dim1=1, center_dim2=1, center_dim3=1, len_dim1=1,
                                 len_dim2=3, len_dim3=1)
    except RuntimeError:
        # If an exception occurs, this is the desired behaviour since the mask is the wrong dimensions
        return 0

    # If there isn't an error, then there is a problem
    print('\n2D input data does not throw an error')
    assert False
