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
    assert np.linalg.norm(shim_weights) == 1, 'Shim weights are not normalized'
    assert len(shim_weights) == b1_maps.shape[3]


def test_b1_shim_wrong_ndim():
    with pytest.raises(ValueError, match=r"Unexpected negative magnitude values"):
        b1_shim(b1_maps[:, :, :, 0], mask)


def test_b1_shim_wrong_mask_shape():
    with pytest.raises(ValueError, match=r"Mask and maps dimensions not matching"):
        b1_shim(b1_maps, mask[:-1, :, :])
