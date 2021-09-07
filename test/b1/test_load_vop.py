#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import os
import pytest
import scipy
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.b1.load_vop import load_siemens_vop

path_sar_file = os.path.join(__dir_testing__, 'b1_maps', 'vop', 'SarDataUser.mat')


def test_load_siemens_vop():
    vop = load_siemens_vop(path_sar_file)
    assert np.isclose(vop[:, 4, 55], [0.00028431 - 2.33700119e-04j,  0.00039449 - 3.11945268e-04j,
                                      0.00052208 - 1.17153693e-03j,  0.00104146 - 1.76284793e-03j,
                                      0.00169108 + 2.29006638e-21j,  0.00051032 + 4.99291087e-04j,
                                      0.0002517 + 2.01207529e-04j, -0.00017224 + 6.93976758e-04j]).all()


def test_load_siemens_vop_wrong_path():
    with pytest.raises(FileNotFoundError, match='The SarDataUser.mat file could not be found.'):
        load_siemens_vop('dummy_path')


def test_load_siemens_vop_no_vop():
    data_no_vop = scipy.io.loadmat(path_sar_file)
    data_no_vop.pop('ZZ')
    path_sar_file_no_vop = os.path.join(__dir_testing__, 'b1_maps', 'vop', 'SarDataUser_no_vop.mat')
    scipy.io.savemat(path_sar_file_no_vop, data_no_vop)
    with pytest.raises(ValueError, match='The SAR data does not contain the expected VOP values.'):
        load_siemens_vop(path_sar_file_no_vop)
    os.remove(path_sar_file_no_vop)
