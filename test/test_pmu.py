#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import numpy as np

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.utils import dicom_times_to_ms


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

    # TODO: Move to appropriate place
    # TODO: possibly input dict (obtained from the json)
    def get_acquisition_times(fname_acquisition):
        """
        Return the acquisition timestamps from a nifti file with corresponding json sidecar. This assumes BIDS
        convention

        Args:
            fname_acquisition (str): Filename corresponding to a nifti file. The file must have a json sidecar with the
                                     same name in the same folder. (nifti.nii nifti.json)
                                     Supported extensions : ".nii", ".nii.gz".

        Returns:
            numpy.ndarray: Acquisition timestamps in ms

        """
        # Get number of volumes
        nii_fieldmap = nib.load(fname_acquisition)
        n_volumes = nii_fieldmap.header['dim'][4]

        # get time between volumes and acquisition start time
        fname, ext = os.path.splitext(fname_acquisition)
        if ext == '.nii':
            fname_acquisition_json = fname + '.json'
        elif ext == '.gz':
            # Disregard gz and test for nii
            fname, ext = os.path.splitext(fname)
            if ext == '.nii':
                fname_acquisition_json = fname + '.json'
            else:
                raise RuntimeError('Input file is not a .nii.gz file')
        else:
            raise RuntimeError('Input file is not a .nii or .XX.gz file')

        json_data = json.load(open(fname_acquisition_json))
        delta_t = json_data['RepetitionTime'] * 1000  # [ms]
        acq_start_time_iso = json_data['AcquisitionTime']  # ISO format
        acq_start_time_ms = dicom_times_to_ms(np.array([acq_start_time_iso]))[0]  # [ms]

        return np.linspace(acq_start_time_ms, ((n_volumes - 1) * delta_t) + acq_start_time_ms, n_volumes)  # [ms]

    # Get the pressure values
    fname_pmu = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
    pmu = PmuResp(fname_pmu)

    # Get acquisition timestamps
    fname_phase_diff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                         'sub-example_phasediff.nii.gz')
    fieldmap_timestamps = get_acquisition_times(fname_phase_diff)

    # Interpolate PMU values onto MRI acquisition timestamp
    acquisition_pressures = pmu.interp_resp_trace(fieldmap_timestamps)

    # Get B0 data
    fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_fieldmap.nii.gz')
    fieldmap = nib.load(fname_fieldmap).get_fdata()

    # Set up mask
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
    ax.set_title("Fieldmap average over unmasked region (Hz) vs time (s)")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'pmu_plot.png')
    fig.savefig(fname_figure)

    # Plot mask
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    im = ax.imshow(fieldmap_masked[:, :, 0, 0])
    fig.colorbar(im)
    ax.set_title("Mask (Hz)")

    ax = fig.add_subplot(212)
    im = ax.imshow(fieldmap[:, :, 0, 0])
    fig.colorbar(im)
    ax.set_title("Fieldmap (Hz)")

    fname_figure = os.path.join(__dir_shimmingtoolbox__, 'mask.png')
    fig.savefig(fname_figure)
