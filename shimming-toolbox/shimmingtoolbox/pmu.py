#!/usr/bin/python3
# -*- coding: utf-8 -*
#
# Read Siemens Physiological Log files
# Adapted from https://gist.github.com/rtrhd/6172344
#

import logging
from matplotlib.figure import Figure
import numpy as np

from shimmingtoolbox.load_nifti import (get_acquisition_start_time, get_acquisition_stop_time, is_fatsat_on,
                                        get_excitation_tr, get_slice_timing, get_slice_tr, get_partial_fourier)

logger = logging.getLogger(__name__)
ERROR_MARGIN = 0.99


class Pmu(object):
    # __stop_time_mdh = None
    # __start_time_mdh = None
    # __data = None
    # timepoints = None
    # data_triggers = None

    def __init__(self, fname_pmu: str):
        attributes = self.read_pmu(fname_pmu)
        self.fname = attributes['fname']
        self.__data = attributes['data']
        self.__start_time_mdh = attributes['start_time_mdh']
        self.__stop_time_mdh = attributes['stop_time_mdh']
        self.start_time_mpcu = attributes['start_time_mpcu']
        self.stop_time_mpcu = attributes['stop_time_mpcu']
        self.data_triggers = attributes['data_triggers']
        self.min = None
        self.max = None
        self.time_offset = 0
        self.timepoints = self.get_all_times()

    def set_data(self, data):
        """
        Set the data of the PMU object

        Args:
            data (numpy.ndarray): Pressure values ranging from 0 to 4095
        """
        self.__data = data
        self.timepoints = self.get_all_times()

    def get_data(self):
        """
        Retrieves the data of the PMU object
        """
        return self.__data

    def set_start_and_stop_times(self, start_time_mdh, stop_time_mdh):
        """
        Set the start and stop time of the PMU object

        Args:
            start_time_mdh (int): Start time in milliseconds past midnight (mdh clock is expected to be the closest to
                                  the image header)
            stop_time_mdh (int): Stop time in milliseconds past midnight (mdh clock is expected to be the closest to
                                 the image header)
        """
        self.__start_time_mdh = start_time_mdh
        self.__stop_time_mdh = stop_time_mdh
        self.timepoints = self.get_all_times()

    def get_start_and_stop_times(self):
        """
        Retrieves the start and stop time of the PMU object
        """
        return self.__start_time_mdh, self.__stop_time_mdh

    def get_all_times(self):
        """
        Get all the timepoints from the respiratory file (in ms).

        Returns:
            np.ndarray: Array containing the timepoints in ms of each data
        """
        raster = float(self.__stop_time_mdh - self.__start_time_mdh) / (len(self.__data) - 1)
        times = (self.__start_time_mdh + raster * np.arange(len(self.__data)))  # ms
        return times

    def get_times(self, start_time=None, stop_time=None):
        """
        Get the times in ms at which the respiration took place.

            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            np.ndarray: Array containing the timepoints in ms of each data
        """
        start_idx, stop_idx = self._get_time_indexes(start_time, stop_time)

        return self.timepoints[start_idx:stop_idx + 1]

    def _get_time_indexes(self, start_time=None, stop_time=None):
        """
        Get the indexes of the data corresponding to the start and stop times

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            tuple: Tuple containing the indexes of the start and stop times
        """
        if start_time is None:
            start_time = self.timepoints[0]
        if stop_time is None:
            stop_time = self.timepoints[-1]

        start_idx = np.argmin(np.abs(self.timepoints - start_time))
        stop_idx = np.argmin(np.abs(self.timepoints - stop_time))

        return start_idx, stop_idx

    def get_trace(self, start_time=None, stop_time=None):
        """
        Returns the trace between ``start_time`` and ``stop_time``

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            numpy.ndarray: Array with the trace between ``start_time`` and ``stop_time``
        """

        start_idx, stop_idx = self._get_time_indexes(start_time, stop_time)
        # +1 is needed to include the stop_idx
        return self.__data[start_idx:stop_idx + 1]

    def read_pmu(self, fname_pmu):
        """
        Read a Siemens Physiological Log file. Returns a tuple with the logging data as numpy integer array and times
        in the form of milliseconds past midnight.

        Args:
            fname_pmu: Filename of the Siemens .resp/.ext file

        Returns:
            dict: A dict containing the ``fname_pmu`` infos. Contains the following keys:

                  * ``fname``
                  * ``data``
                  * ``data_triggers``
                  * ``start_time_mdh``
                  * ``stop_time_mdh``
                  * ``start_time_mpcu``
                  * ``stop_time_mpcu``

        """
        f = None
        try:
            f = fname_pmu if hasattr(fname_pmu, 'read') else open(fname_pmu)
            lines = [line for line in f]
        finally:
            if f:
                f.close()

        # Most values are space separated on a single (the first) line
        fields = lines[0].split(' ')

        if fields[5] != 5002:
            # non cardiac gate close seems to be missing
            clean_fields = []
            start_field = 4
        else:
            # cardiac has gate closed and 5002 info field
            clean_fields = fields[5]
            start_field = 5

        # Remove anything bracketed by 5002, 6002 (Information Fields)
        include = True
        for field in fields[start_field:]:
            if include:
                if field == '5002':
                    include = False
                else:
                    clean_fields.append(field)
            else:
                if field == '6002':
                    include = True

        # Drop the '5003\r\n' or '6003\r\n' at the end
        clean_fields = clean_fields[:-1]

        # Returned values will be a numpy array of ints
        data = np.asarray([int(field) for field in clean_fields])

        # Extract the start/stop times from subsequent lines
        start_time_mdh = int([line for line in lines if line.startswith('LogStartMDHTime')][0].split()[1])
        stop_time_mdh = int([line for line in lines if line.startswith('LogStopMDHTime')][0].split()[1])
        start_time_mpcu = int([line for line in lines if line.startswith('LogStartMPCUTime')][0].split()[1])
        stop_time_mpcu = int([line for line in lines if line.startswith('LogStopMPCUTime')][0].split()[1])

        # Get rid of the 5000 and 6000 trigger marks from the data. Also, in the case of ECG log files
        # we'll also have multiple traces in the file. The second trace will be offset by 8192 so for ECG
        # files limiting the data to points in the range 0..4095 will give us just the first channel's data.
        # We are assuming that the 5000/6000 marks are inserted into the data values list rather than overwriting.
        # That is we assume they do *not* occupy a raster position.
        data_cleaned = data[data < 4096]

        attributes = {
            'fname': fname_pmu,
            'data': data_cleaned,
            'data_triggers': data,
            'start_time_mdh': start_time_mdh,
            'stop_time_mdh': stop_time_mdh,
            'start_time_mpcu': start_time_mpcu,
            'stop_time_mpcu': stop_time_mpcu,
        }

        return attributes

    def get_trigger_times(self, start_time=None, stop_time=None):
        """
        Returns the trigger times in ms of the resp trace. These triggers estimate the beginning of a new respiratory
        cycle

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            numpy.ndarray: Array with the trigger times in ms of the resp trace
        """
        index_start, index_stop = self._get_time_indexes(start_time, stop_time)

        trigger_times = []
        i = 0
        for data in self.data_triggers:
            if data > 4096:
                if data == 5000 and (index_start <= i <= index_stop):
                    trigger_times.append(float(self.timepoints[i]))
            else:
                i += 1

        logger.debug(f"Trigger times: {trigger_times}")

        return np.array(trigger_times)

    def get_mean_trigger_span(self):
        """
        Returns the mean time between triggers in ms

        Returns:
            float: Mean time between triggers in ms
        """
        trigger_times = self.get_trigger_times()
        trigger_span = np.diff(trigger_times)
        return np.mean(trigger_span)

    def plot_triggers(self, fname_output, start_time=None, stop_time=None):
        trigger_times = self.get_trigger_times(start_time=start_time, stop_time=stop_time)
        times = self.get_times(start_time=start_time, stop_time=stop_time)
        if len(trigger_times) == 0:
            raise ValueError("No trigger times found in the specified range.")

        fig = Figure(figsize=(8, 4), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        for trigger_time in trigger_times:
            ax.vlines((trigger_time - min(times)) / 1000, 0, 1, color='b', linestyle='-')
        ax.hlines(0, 0, (max(times) - min(times)) / 1000, color='k', linestyle='-')

        ax.set(xlim=(0, (max(times) - min(times)) / 1000), ylim=(-0.05, 1.05))

        ax.set_title(f"Triggers through time, # of triggers: {len(trigger_times)}, avg time between triggers: {self.get_mean_trigger_span() / 1000:.3f} s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("a.u.")

        fig.savefig(fname_output, bbox_inches='tight')


class PmuResp(Pmu):
    """
    PMU object containing the pressure values of a Siemens .resp file

    Attributes:
            fname (str): Filename of the Siemens .resp file
            data (numpy.ndarray): Pressure values ranging from 0 to 4095
            start_time_mdh (int): Start time in milliseconds past midnight (mdh clock is expected to be the closest to
                                  the image header)
            stop_time_mdh (int): Stop time in milliseconds past midnight (mdh clock is expected to be the closest to
                                 the image header)
            start_time_mpcu (int): Start time in milliseconds past midnight
            stop_time_mpcu (int): Stop time in milliseconds past midnight
    """
    def __init__(self, fname_pmu: str, time_offset=0):
        """

        Args:
            fname_pmu (str): Filename of the Siemens .resp file
            time_offset (int): Time offset in ms to what is read in the .resp file
        """
        super().__init__(fname_pmu)
        self.min = 0
        self.max = 4095
        self.adjust_start_time(time_offset)
        self.timepoints = self.get_all_times()

    def adjust_start_time(self, time_offset: int):
        """
        Offset the start and end time of the PMU data
        Args:
            time_offset (int): Time offset in ms to what is read in the .resp file

        """
        old_offset = self.time_offset
        start_time, stop_time = self.get_start_and_stop_times()
        self.set_start_and_stop_times(start_time + time_offset - old_offset, stop_time + time_offset - old_offset)
        self.start_time_mpcu += time_offset - old_offset
        self.stop_time_mpcu += time_offset - old_offset
        self.time_offset = time_offset

    def interp_resp_trace(self, acquisition_times):
        """
        Interpolates ``data`` to the specified ``acquisition_times``

        Args:
            acquisition_times (numpy.ndarray): Array of the times in milliseconds past midnight of the desired
                                               times to interpolate the resp_trace. Times must be within
                                               ``self.__start_time_mdh`` and ``self.__stop_time_mdh``

        Returns:
            numpy.ndarray: Array with interpolated times with the same shape as ``acquisition_times``
        """
        start_time, stop_time = self.get_start_and_stop_times()
        if np.any(start_time > acquisition_times) or np.any(stop_time < acquisition_times):
            start_offset = np.min(acquisition_times - start_time)
            stop_offset = np.min(stop_time - acquisition_times)
            logger.debug(f"start_offset: {start_offset}")
            logger.debug(f"stop_offset: {stop_offset}")
            raise OutOfRangeError("acquisition_times don't fit within time limits for resp trace")

        interp_data = np.interp(acquisition_times, self.get_times(), self.get_data())

        return interp_data

    def mean(self, start_time=None, stop_time=None):
        """
        Returns the mean value of the resp trace between ``start_time`` and ``stop_time``

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            float: Mean value of the resp trace between ``start_time`` and ``stop_time``
        """

        pressures = self.get_trace(start_time=start_time, stop_time=stop_time)

        return np.mean(pressures)

    def get_pressure_rms(self, start_time=None, stop_time=None):
        """
        Returns the RMS value of the resp trace between ``start_time`` and ``stop_time``

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            float: RMS value of the resp trace between ``start_time`` and ``stop_time``
        """

        pressures = self.get_trace(start_time=start_time, stop_time=stop_time)
        mean_p = self.mean(start_time=start_time, stop_time=stop_time)

        return np.sqrt(np.mean((pressures - mean_p)**2))


class PmuExt(Pmu):
    def __init__(self, fname_pmu: str):
        """
        PMU object containing the trigger values of a Siemens .trig file

        Args:
            fname_pmu (str): Filename of the Siemens .trig file
        """
        super().__init__(fname_pmu)
        self.min = 0
        self.max = 1

    def get_acquisition_times(self, nii_data, json_data, delay=2):
        """ Returns the time at which the middle of the slice was acquired based on getting a trigger at the start of
            a slice. That is: slice_middle = trigger_time + rf_to_middle_of_slice + delay.

        Args:
            nii_data (nibabel.Nifti1Image): Nibabel object containing the image timeseries.
            json_data (dict): Json dict corresponding to a nifti sidecar (BIDS format).
            delay (int): Delay in ms to add to the trigger time. There is a 2ms delay between the trigger and the RF
                         pulse.

        Returns:
            numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
        """

        if len(nii_data.shape) == 4:
            n_volumes = nii_data.shape[3]
        else:
            n_volumes = 1

        if len(nii_data.shape) != 2:
            n_slices = nii_data.shape[2]
        else:
            n_slices = 1

        # Can be '2D'
        mr_acquisition_type = json_data.get('MRAcquisitionType')
        if mr_acquisition_type != '2D':
            # mr_acquisition_type is None or 3D
            raise NotImplementedError("MR acquisition type is not 2D.")

        # Offset time in ms to add a buffer to the imprecise timing of the DICOMS
        offset = 5000
        acq_start_time = int(get_acquisition_start_time(json_data)) - offset
        acq_stop_time = int(get_acquisition_stop_time(nii_data, json_data)) + offset

        trigger_times = self.get_trigger_times(acq_start_time, acq_stop_time)
        if len(trigger_times) == 0:
            raise ValueError("No trigger times found in the specified range.")

        fat_suppression = is_fatsat_on(json_data)

        # If fat suppression, discard half the triggers
        if fat_suppression:
            # If there is fat sat, the 2nd, 4th, ... times correspond to a fat sat trigger
            # We only keep the imaging triggers (i.e.: 1st, 3rd, ... -> index 0, 2, 4...)
            trigger_times = trigger_times[::2]

        # Make sure we have the expected number of triggers
        if len(trigger_times) != n_volumes * n_slices:
            raise ValueError("Not enough trigger times for the specified number of volumes and slices.")

        slice_timing_start = get_slice_timing(json_data, n_slices)

        repetition_slice_excitation = get_excitation_tr(json_data)

        # If the slice timing is lower than the TR excitation, this is an interleaved multi-slice acquisition
        # (more than one slice acquired within a TR excitation)
        # Remove slices that are at 0 ms (acquired during the first TR excitation)
        # If there are other slices acquired before the next TR excitation, then this is interleaved
        # multi-slice
        if np.any(slice_timing_start[slice_timing_start > 0] < repetition_slice_excitation):
            raise NotImplementedError("Interleaved multi-slice acquisition detected, but not implemented.")

        deltat_slice = get_slice_tr(json_data)

        # Order according to slice timing
        slice_timing_from_triggers = np.zeros(n_volumes * n_slices, dtype=float)
        slice_timing_order = np.argsort(slice_timing_start)
        for i_vol in range(n_volumes):
            slice_timing_from_triggers[i_vol * n_slices:(i_vol + 1) * n_slices] = trigger_times[i_vol * n_slices:(i_vol + 1) * n_slices][slice_timing_order]

        # Get when the middle of k-space was acquired
        fourier = get_partial_fourier(json_data)

        # The crossing of the center-line of k-space in the phase encoding direction is not exactly at 1/2
        # of the time it takes to acquire a slice if partial Fourier is not 1. For a 7/8 partial Fourier,
        # it would acquire the center-line of k-space after 3/7 of the time it takes to acquire a slice.
        # (For a 700ms slice, the center-line would be acquired ~300ms). I simulated a couple of sequences in
        # POET and observed that the "smaller" portion is always acquired first (the 1/8 portion is skipped,
        # then 3/8 is acquired, k-space is crossed then the final 1/2 is acquired. In total, the k-space crossing
        # was at 3/7 of the slice time).
        ratio = (fourier - 0.5) / fourier

        # Calculate acquisition timestamps
        acquisition_timestamps = slice_timing_from_triggers + (deltat_slice * ratio) + delay

        # Reshape to (n_volumes, n_slices)
        acquisition_timestamps = acquisition_timestamps.reshape(n_volumes, n_slices)

        return acquisition_timestamps


class OutOfRangeError(Exception):
    """
    Exception raised when the requested time is out of range of the PMU data.
    """
