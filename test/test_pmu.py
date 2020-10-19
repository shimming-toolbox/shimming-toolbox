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
import nibabel as nib
import json
from shimmingtoolbox.masking.shapes import shapes


def test_timing_images():
    """Check the matching of timing between MR images and PMU timestamps"""

    # Convert to ms
    def dicom_times_to_ms(dicom_times):
        """
        Convert dicom acquisition times to ms

        Args:
            dicom_times (numpy.ndarray): 1D array of time strings from dicoms.
                                         Suported formats: "HHMMSS.mmmmmm" or "HH:MM:SS.mmmmmm

        Returns:
            numpy.ndarray: 1D array of times in milliseconds
        """

        ms_times = []

        for a_time in dicom_times:
            if len(a_time) == 13 and a_time[6] == '.' and isinstance(a_time, str):
                hours = int(a_time[0:2])
                minutes = int(a_time[2:4])
                seconds = int(a_time[4:6])
                micros = int(a_time[7:13])
            elif len(a_time) == 15 and a_time[2] + a_time[5] + a_time[8] == ['::.'] or isinstance(a_time, str):
                hours = int(a_time[0:2])
                minutes = int(a_time[3:5])
                seconds = int(a_time[6:8])
                micros = int(a_time[9:15])
            else:
                raise RuntimeError("Input format does not follow 'HHMMSS.mmmmmm'")

            ms_times.append(1000 * (hours * 3600 + minutes * 60 + seconds) + micros / 1000)  # ms

        return np.array(ms_times)

    fname_pmu = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_pmu)

    # TODO: Update testing data with the updated niftis processed by the new version of dcm2niix
    #  (Note: the appropriate version is currently the dev version as of oct 16 2020)
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_fieldmap.nii.gz')
    fname_json_phase_diff = os.path.join(__dir_testing__, 'nifti', 'sub-example', 'fmap',
                              'sub-example_phasediff.json')

    # get time between volumes and acquisition start time
    json_data = json.load(open(fname_json_phase_diff))
    # TODO: TimeBetweenVolumes will most likely be changed to repetitionTime eventually according to
    #  https://github.com/UNFmontreal/Dcm2Bids/issues/90
    delta_t = json_data['TimeBetweenVolumes'] * 1000  # [ms]
    acq_start_time = json_data['AcquisitionTime']  # ISO format
    acq_start_time = dicom_times_to_ms(np.array([acq_start_time]))[0]  # [ms]

    # Get the number of volumes
    nii_fieldmap = nib.load(fname_fieldmap)
    n_volumes = nii_fieldmap.header['dim'][4]
    fieldmap_timestamps = np.linspace(acq_start_time, ((n_volumes - 1) * delta_t) + acq_start_time, n_volumes)

    # These timestamps were generated as explained here: https://github.com/UNFmontreal/Dcm2Bids/issues/90
    # in microseconds
    # data_timestamps = ['121821.960000', '121822.745000', '121816.452500', '121817.240000', '121818.025000',
    #                    '121821.172500', '121820.385000', '121818.812500', '121819.600000', '121823.532500']

    acquisition_pressures = pmu.interp_resp_trace(fieldmap_timestamps)

    # Get B0 data
    fieldmap = nii_fieldmap.get_fdata()
    mask_len1 = 15
    mask_len2 = 5
    mask_len3 = fieldmap.shape[2]
    mask = shapes(fieldmap[:, :, :, 0], shape='cube',
                  center_dim1=int(fieldmap.shape[0] / 2 - 8),
                  center_dim2=int(fieldmap.shape[1] / 2 - 20),
                  len_dim1=mask_len1, len_dim2=mask_len2, len_dim3=mask_len3)

    fieldmap_masked = np.zeros_like(fieldmap)
    fieldmap_mean = np.zeros([fieldmap.shape[3]])
    for i_time in range(fieldmap.shape[3]):
        fieldmap_masked[:, :, :, i_time] = fieldmap[:, :, :, i_time] * mask
        masked_array = np.ma.array(fieldmap[:, :, :, i_time], mask=mask == False)
        fieldmap_mean[i_time] = np.ma.average(masked_array)

    # Sanity check -->
    pmu_times = np.linspace(pmu.start_time_mdh, pmu.stop_time_mdh, len(pmu.data))
    pmu_times_within_range = pmu_times[pmu_times > fieldmap_timestamps[0]]
    pmu_data_within_range = pmu.data[pmu_times > fieldmap_timestamps[0]]
    pmu_data_within_range = pmu_data_within_range[pmu_times_within_range < fieldmap_timestamps[fieldmap.shape[3] - 1]]
    pmu_times_within_range = pmu_times_within_range[pmu_times_within_range < fieldmap_timestamps[fieldmap.shape[3] - 1]]

    # Plot results
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    ax.plot(fieldmap_timestamps / 1000, acquisition_pressures, label='Interpolated pressures')
    # ax.plot(pmu_times, pmu.data, label='Raw pressures')
    ax.plot(pmu_times_within_range / 1000, pmu_data_within_range, label='Pmu pressures')
    ax.legend()
    ax.set_title("Pressure [-2048, 2047] vs time (s) ")
    ax = fig.add_subplot(212)
    ax.plot(fieldmap_timestamps / 1000, fieldmap_mean, label='Mean B0')
    ax.legend()
    ax.set_title("Fieldmap average over unmasked region (hz) vs time (s)")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'pmu_plot.png')
    fig.savefig(fname_figure)

    # Plot mask
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    im = ax.imshow(fieldmap_masked[:, :, 0, 0])
    fig.colorbar(im)
    ax.set_title("Mask (hz)")

    ax = fig.add_subplot(212)
    im = ax.imshow(fieldmap[:, :, 0, 0])
    fig.colorbar(im)
    ax.set_title("Fieldmap (hz)")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'mask.png')
    fig.savefig(fname_figure)