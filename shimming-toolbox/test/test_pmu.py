#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import numpy as np
import nibabel as nib
import json
import scipy.signal

from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.load_nifti import get_acquisition_times
from shimmingtoolbox.files.NiftiFieldMap import NiftiFieldMap

fname_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime', 'sub-realtime_PMUresp_signal.resp')
nif_fieldmap = NiftiFieldMap(fname_fieldmap, dilation_kernel_size=3, is_realtime=True)
# Get the pressure values
pmu = PmuResp(fname_resp)


def test_read_resp():
    expected_first_data = np.array([1667, 1667, 1682, 1682, 1667]) + 2048
    expected_last_data = np.array([-2048, -2048, -2048, -2048, -2048]) + 2048
    expected_start_time_mdh = 44294387
    expected_stop_time_mdh = 44343130
    expected_start_time_mpcu = 44294295
    expected_stop_time_mpcu = 44343040

    data = pmu.get_data()
    start_time_mdh, stop_time_mdh = pmu.get_start_and_stop_times()
    assert np.all([
        np.all(expected_first_data == data[:5]),
        np.all(expected_last_data == data[-5:]),
        expected_start_time_mdh == start_time_mdh,
        expected_stop_time_mdh == stop_time_mdh,
        expected_start_time_mpcu == pmu.start_time_mpcu,
        expected_stop_time_mpcu == pmu.stop_time_mpcu
    ])


def test_interp_resp_trace():
    # Create time series to interpolate the PMU to
    num_points = 20
    start_time_mdh, stop_time_mdh = pmu.get_start_and_stop_times()
    acq_times = np.linspace(start_time_mdh, stop_time_mdh, num_points)

    acq_pressure = pmu.interp_resp_trace(acq_times)
    data = pmu.get_data()
    
    index_pmu_data = np.linspace(0, len(data) - 1, int(num_points)).astype(int)
    index_pmu_interp = np.linspace(0, num_points - 1, int(num_points)).astype(int)

    assert(np.all(np.isclose(acq_pressure[index_pmu_interp], data[index_pmu_data], atol=1, rtol=0.08)))


def test_timing_images():
    """Check the matching of timing between MR images and PMU timestamps"""
    # Get acquisition timestamps
    fname_phase_diff_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap',
                                         'sub-realtime_phasediff.json')
    with open(fname_phase_diff_json) as json_file:
        json_data = json.load(json_file)
    nif_fieldmap.json = json_data
    fieldmap_timestamps = get_acquisition_times(nif_fieldmap)

    # Interpolate PMU values onto MRI acquisition timestamp
    acquisition_pressures = pmu.interp_resp_trace(fieldmap_timestamps)

    # Set up mask
    mask_len1 = 15
    mask_len2 = 5
    mask_len3 = nif_fieldmap.shape[2]
    mask = shapes(nif_fieldmap.data[:, :, :, 0], shape='cube',
                  center_dim1=int(nif_fieldmap.shape[0] / 2 - 8),
                  center_dim2=int(nif_fieldmap.shape[1] / 2 - 20),
                  len_dim1=mask_len1, len_dim2=mask_len2, len_dim3=mask_len3)

    # Apply mask and compute the average for each timepoint
    fieldmap_masked = np.zeros_like(nif_fieldmap.data)
    fieldmap_avg = np.zeros([nif_fieldmap.shape[3]])
    for i_time in range(nif_fieldmap.shape[3]):
        fieldmap_masked[:, :, :, i_time] = nif_fieldmap.data[:, :, :, i_time] * mask
        masked_array = np.ma.array(nif_fieldmap.data[:, :, :, i_time], mask=mask == False)
        fieldmap_avg[i_time] = np.ma.average(masked_array)

    # Compute correlation
    pearson = np.corrcoef(fieldmap_avg, acquisition_pressures[:, 0])

    assert(np.isclose(pearson[0, 1], 0.97095326))


def test_pmu_fake_data():
    fake_data = np.array([3000, 2000, 1000, 2000, 3000, 2000, 1000, 2000, 3000, 2000])
    pmu.set_data(fake_data)
    pmu.set_start_and_stop_times(125, 250 * (len(fake_data) - 1) + 125)

    json_data = {'RepetitionTime': 250 / 1000, 'AcquisitionTime': "00:00:00.000000"}
    nif_fieldmap.json = json_data
    # Calc pressure
    acq_timestamps = get_acquisition_times(nif_fieldmap)
    acq_pressures = pmu.interp_resp_trace(acq_timestamps)

    # 10 volumes, 1 slice
    assert np.all(acq_pressures[:, 0] == fake_data)
