#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import numpy as np

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.pmu import PmuResp


def test_read_resp():
    fname = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname)

    expected_first_data = np.array([1667, 1667, 1682, 1682, 1667])
    expected_last_data = np.array([-2048, -2048, -2048, -2048, -2048])
    expected_start_time_mdh = 44294387
    expected_stop_time_mdh = 44343130
    expected_start_time_mpcu = 44294295
    expected_stop_time_mpcu = 44343040

    assert np.all([
        np.all(expected_first_data == pmu.data[:5]),
        np.all(expected_last_data == pmu.data[-5:]),
        expected_start_time_mdh == pmu.start_time_mdh,
        expected_stop_time_mdh == pmu.stop_time_mdh,
        expected_start_time_mpcu == pmu.start_time_mpcu,
        expected_stop_time_mpcu == pmu.stop_time_mpcu
    ])


def test_interp_resp_trace():
    fname = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname)

    # Create time series to interpolate the PMU to
    num_points = 20
    acq_times = np.linspace(pmu.start_time_mdh, pmu.stop_time_mdh, num_points)

    acq_pressure = pmu.interp_resp_trace(acq_times)

    index_pmu_data = np.linspace(0, len(pmu.data) - 1, int(num_points)).astype(int)
    index_pmu_interp = np.linspace(0, num_points - 1, int(num_points)).astype(int)

    assert(np.all(np.isclose(acq_pressure[index_pmu_interp], pmu.data[index_pmu_data], atol=1, rtol=0.08)))

