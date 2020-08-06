# coding: utf-8

import numpy as np
import pytest

import shimmingtoolbox as shim
import shimmingtoolbox.masking.threshold


dummy_data = [
    (np.array([3, 4]), np.array([False,  True])),
    (np.array([[3, 4], [5, 6]]), np.array([[False,  True], [True, True]])),
]


@pytest.mark.parametrize('data,expected', dummy_data)
def test_threshold(data, expected):
    assert np.all(shim.masking.threshold.threshold(data, thr=3) == expected)
