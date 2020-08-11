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
    (np.array([[3, 4], [5, 6], [7, 8]]), 'square'),
]
dummy_data_shape_cube = [
    (np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
               [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
               [[1, 2, 3], [4, 5, 6], [7, 8, 11]]]), 'cube'),
]


@pytest.mark.parametrize('data,expected', dummy_data)
def test_threshold(data, expected):
    assert np.all(shim.masking.threshold.threshold(data, thr=3) == expected)


@pytest.mark.parametrize('data,shape', dummy_data_shape_square)
def test_mask_square(data, shape):
    mask = shim.masking.shape.shape(data, shape, center_x=0, center_y=1, len_x=1, len_y=3)
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(mask[:, :])
    ax.set_title("Mask")
    fig.savefig("mask.png")


@pytest.mark.parametrize('data,shape', dummy_data_shape_cube)
def test_mask_cube(data, shape):
    mask = shim.masking.shape.shape(data, shape, center_x=0, center_y=1, center_z=1, len_x=1, len_y=3, len_z=1)
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(mask[:, :, 1])
    ax.set_title("Mask")
    fig.savefig("mask.png")
