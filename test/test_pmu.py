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


from matplotlib.figure import Figure
from shimmingtoolbox import __dir_shimmingtoolbox__


def test_timing_images():
    """Check the matching of timing between MR images and PMU timestamps"""

    fname_pmu = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_pmu)
    fname_data = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap', '')

    # These timestamps were generated as explained here: https://github.com/UNFmontreal/Dcm2Bids/issues/90
    # in microseconds
    data_timestamps = ['121821.960000', '121822.745000', '121816.452500', '121817.240000', '121818.025000',
                       '121821.172500', '121820.385000', '121818.812500', '121819.600000', '121823.532500']

    # Convert to ms
    def dicom_times_to_ms(dicom_times):
        """
        Convert dicom acquisition times to ms

        Args:
            dicom_times (numpy.ndarray): 1D array of time strings from dicoms. Format: "HHMMSS.mmmmmm"

        Returns:
            numpy.ndarray: 1D array of times in milliseconds
        """

        ms_times = []

        for a_time in dicom_times:
            if len(a_time) != 13 or a_time[6] != '.' or not isinstance(a_time, str):
                raise RuntimeError("Input format does not follow 'HHMMSS.mmmmmm'")
            hours = int(a_time[0:2])
            minutes = int(a_time[2:4])
            seconds = int(a_time[4:6])
            micros = int(a_time[7:13])

            ms_times.append(1000 * (hours * 3600 + minutes * 60 + seconds) + micros / 1000)  # ms

        return ms_times

    acquisition_times = dicom_times_to_ms(data_timestamps)

    acquisition_times = sorted(acquisition_times)
    acquisition_times = np.array(acquisition_times)

    acquisition_pressures = pmu.interp_resp_trace(acquisition_times)

    # Sanity check -->
    pmu_times = np.linspace(pmu.start_time_mdh, pmu.stop_time_mdh, len(pmu.data))

    # Plot results
    fig = Figure(figsize=(10, 10))
    # FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(acquisition_times, acquisition_pressures)
    ax.plot(pmu_times, pmu.data)
    ax.set_title("test")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'pmu_plot.png')
    fig.savefig(fname_figure)

