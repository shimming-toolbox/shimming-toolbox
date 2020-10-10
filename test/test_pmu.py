#!/usr/bin/python3
# -*- coding: utf-8 -*

import os

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.pmu import read_resp
import numpy as np


def test_read_resp():
    fname = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    data_cleaned, start_time_mdh, stop_time_mdh, start_time_mpcu, stop_time_mpcu = read_resp(fname)

    expected_first_data = np.array([1667, 1667, 1682, 1682, 1667])
    expected_last_data = np.array([-2048, -2048, -2048, -2048, -2048])
    expected_start_time_mdh = 44294387
    expected_stop_time_mdh = 44343130
    expected_start_time_mpcu = 44294295
    expected_stop_time_mpcu = 44343040

    assert np.all([
        np.all(expected_first_data == data_cleaned[:5]),
        np.all(expected_last_data == data_cleaned[-5:]),
        expected_start_time_mdh == start_time_mdh,
        expected_stop_time_mdh == stop_time_mdh,
        expected_start_time_mpcu == start_time_mpcu,
        expected_stop_time_mpcu == stop_time_mpcu
    ])


