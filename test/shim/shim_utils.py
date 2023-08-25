#!usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import pytest

from shimmingtoolbox.shim.shim_utils import convert_to_mp, phys_to_shim_cs, shim_to_phys_cs


def test_convert_to_mp_unknown_scanner(caplog):
    dac_units = [14436, 14265, 14045, 9998, 9998, 9998, 9998, 9998]

    convert_to_mp(dac_units, 'unknown')
    assert "Manufacturer unknown not implemented, bounds might not be respected. Setting initial " \
           "shim_setting to 0" in caplog.text


def test_convert_to_mp_outside_bounds():
    dac_units = [20000, 14265, 14045, 9998, 9998, 9998, 9998, 9998]

    with pytest.raises(ValueError, match="Multipole values exceed known system limits."):
        convert_to_mp(dac_units, 'Prisma_fit')


def test_phys_to_shim_cs():
    out = phys_to_shim_cs(np.array([1, 1, 1]), 'Siemens')
    assert np.all(out == [-1, 1, -1])


def test_shim_to_phys_cs():
    out = shim_to_phys_cs(np.array([1, 1, 1]), 'Siemens')
    assert np.all(out == [-1, 1, -1])
