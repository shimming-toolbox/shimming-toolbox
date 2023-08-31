#!usr/bin/env python3
# -*- coding: utf-8

import logging
import numpy as np
import pytest

from shimmingtoolbox.shim.shim_utils import convert_to_mp, phys_to_shim_cs, shim_to_phys_cs, logger


def test_convert_to_mp_unknown_scanner(caplog):
    dac_units = {'order1': [14436, 14265, 14045], 'order2': [9998, 9998, 9998, 9998, 9998]}

    with caplog.at_level(logging.DEBUG, logger.name):
        convert_to_mp('unknown', dac_units)

    assert "Manufacturer model unknown not implemented, could not convert shim settings" in caplog.text


def test_convert_to_mp_outside_bounds():
    dac_units = {'order1': [20000, 14265, 14045], 'order2': [9998, 9998, 9998, 9998, 9998]}

    with pytest.raises(ValueError, match="Multipole values exceed known system limits."):
        convert_to_mp('Prisma_fit', dac_units)


def test_phys_to_shim_cs():
    out = phys_to_shim_cs(np.array([1, 1, 1]), 'Siemens')
    assert np.all(out == [-1, 1, -1])


def test_shim_to_phys_cs():
    out = shim_to_phys_cs(np.array([1, 1, 1]), 'Siemens')
    assert np.all(out == [-1, 1, -1])
