#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import os
import pytest
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.b1.b1_shim import b1_shim, combine_maps, cov, vector_to_complex
from shimmingtoolbox.load_nifti import read_nii

fname_b1 = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'sub-01_run-10_TB1map.nii.gz')
_, _, b1_maps = read_nii(fname_b1)
mask = b1_maps[:, :, :, 0] != 0


def test_b1_shim():
    shim_weights = b1_shim(b1_maps, mask)
    assert np.linalg.norm(shim_weights) == 1, 'The shim weights are not normalized'
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_wrong_ndim():
    with pytest.raises(ValueError, match=r"Unexpected negative magnitude values."):
        b1_shim(b1_maps[:, :, :, 0], mask)


def test_b1_shim_wrong_mask_shape():
    with pytest.raises(ValueError, match=r"Mask and maps dimensions not matching."):
        b1_shim(b1_maps, mask[:-1, :, :])


def test_vector_to_complex():
    assert np.isclose(vector_to_complex(np.asarray([1, 1, 1, 0, np.pi/2, np.pi])), np.asarray([1,  1j, -1])).all(),\
        "The function vector_to_complex returns unexpected results"


def test_vector_to_complex_wrong_length():
    with pytest.raises(ValueError, match=r"The vector must have an even number of elements."):
        vector_to_complex(np.asarray([1, 1, 1, 0, np.pi/2])), np.asarray([1,  1j, -1])


def test_combine_maps():
    dummy_weights = [1+8j, 3-5j, 8+1j, 4-4j, 5-6j, 2-9j, 3+2j, 4-7j]
    combined_map = combine_maps(b1_maps, dummy_weights)
    values = [combined_map[30, 30, 15], combined_map[20, 30, 7], combined_map[40, 15, 10], combined_map[30, 60, 7]]
    values_to_match = [91.4696493618864, 306.7951929580235, 396.57494585161345, 212.46572478710647]
    assert np.isclose(values, values_to_match).all()


def test_combine_maps_wrong_weights_number():
    dummy_weights = [1+8j, 3-5j, 8+1j, 4-4j, 5-6j, 2-9j, 3+2j]
    with pytest.raises(ValueError, match=r"The number of shim weights does not match the number of coils."):
        combine_maps(b1_maps, dummy_weights)
