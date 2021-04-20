#!usr/bin/env python3
# -*- coding: utf-8

import pytest
from shimmingtoolbox.b1.load_vop import load_vop

path_sar_file = '/Users/gaspard/Desktop/Matlab_files/SarDataUser.mat'


def test_load_vop():
    VOP = load_vop(path_sar_file)


def test_load_vop_wrong_path():
    with pytest.raises(FileNotFoundError, match='The SarDataUser.mat file could not be found.'):
        load_vop('dummy_path')
