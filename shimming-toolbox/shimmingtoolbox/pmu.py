#!/usr/bin/python3
# -*- coding: utf-8 -*
#
# Read Siemens Physiological Log files
# Adapted from https://gist.github.com/rtrhd/6172344
#

from pydicom import dcmread
import gzip
import logging
from matplotlib.figure import Figure
import numpy as np
import os

from shimmingtoolbox.files.NiftiFile import NiftiFile

logger = logging.getLogger(__name__)
ERROR_MARGIN = 0.99


class Pmu(object):
    # _stop_time_mdh = None
    # _start_time_mdh = None
    # _data = None
    # timepoints = None
    # data_triggers = None

    def __init__(self, fname_pmu: str):
        attributes = self.read_pmu(fname_pmu)
        self.fname = attributes['fname']
        self._data = attributes['data']
        self._start_time_mdh = attributes['start_time_mdh']
        self._stop_time_mdh = attributes['stop_time_mdh']
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
        self._data = data
        self.timepoints = self.get_all_times()

    def get_data(self):
        """
        Retrieves the data of the PMU object
        """
        return self._data

    def set_start_and_stop_times(self, start_time_mdh, stop_time_mdh):
        """
        Set the start and stop time of the PMU object

        Args:
            start_time_mdh (int): Start time in milliseconds past midnight (mdh clock is expected to be the closest to
                                  the image header)
            stop_time_mdh (int): Stop time in milliseconds past midnight (mdh clock is expected to be the closest to
                                 the image header)
        """
        self._start_time_mdh = start_time_mdh
        self._stop_time_mdh = stop_time_mdh
        self.timepoints = self.get_all_times()

    def get_start_and_stop_times(self):
        """
        Retrieves the start and stop time of the PMU object
        """
        return self._start_time_mdh, self._stop_time_mdh

    def get_all_times(self):
        """
        Get all the timepoints from the respiratory file (in ms).

        Returns:
            np.ndarray: Array containing the timepoints in ms of each data
        """
        raster = float(self._stop_time_mdh - self._start_time_mdh) / (len(self._data) - 1)
        times = (self._start_time_mdh + raster * np.arange(len(self._data)))  # ms
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
        return self._data[start_idx:stop_idx + 1]

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
        Returns the trigger times in ms of the resp trace

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
            if data > self.max:
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

    def plot_data(self, fname_output, start_time=None, stop_time=None):
        times = self.get_times(start_time, stop_time)

        fig = Figure(figsize=(8, 4), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(times, self.get_data())

        ax.set_title("PMU data over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("a.u.")

        fig.savefig(fname_output, bbox_inches='tight')

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
        self.time_offset = 0
        self.adjust_start_time(time_offset)

    def adjust_start_time(self, time_offset: int):
        """
        Offset the start and end time of the PMU data
        Args:
            time_offset (int): Time offset in ms to what is read in the .resp file

        """
        old_offset = self.time_offset
        start_time, stop_time = self.get_start_and_stop_times()
        self.set_start_and_stop_times(start_time + time_offset - old_offset, stop_time + time_offset - old_offset)
        if self.start_time_mpcu is not None:
            self.start_time_mpcu += time_offset - old_offset
        if self.stop_time_mpcu is not None:
            self.stop_time_mpcu += time_offset - old_offset
        self.time_offset = time_offset

        self.timepoints = self.get_all_times()

    def interp_resp_trace(self, acquisition_times):
        """
        Interpolates ``data`` to the specified ``acquisition_times``

        Args:
            acquisition_times (numpy.ndarray): Array of the times in milliseconds past midnight of the desired
                                               times to interpolate the resp_trace. Times must be within
                                               ``self._start_time_mdh`` and ``self._stop_time_mdh``

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

    def interp_resp_derivative(self, acquisition_times):
        start_time, stop_time = self.get_start_and_stop_times()
        if np.any(start_time > acquisition_times) or np.any(stop_time < acquisition_times):
            start_offset = np.min(acquisition_times - start_time)
            stop_offset = np.min(stop_time - acquisition_times)
            logger.debug(f"start_offset: {start_offset}")
            logger.debug(f"stop_offset: {stop_offset}")
            raise OutOfRangeError("acquisition_times don't fit within time limits for resp trace")

        deriv = np.gradient(self.get_data(), self.get_times())
        interp_data = np.interp(acquisition_times, self.get_times(), deriv)

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

    def get_acquisition_times(self, nif, delay=2):
        """ Returns the time at which the middle of the slice was acquired based on getting a trigger at the start of
            a slice. That is: slice_middle = trigger_time + rf_to_middle_of_slice + delay.

        Args:
            nif (NiftiFile): NiftiFile object containing the image timeseries.
            delay (int): Delay in ms to add to the trigger time. There is a 2ms delay between the trigger and the RF
                         pulse.

        Returns:
            numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
        """

        if nif.ndim == 4:
            n_volumes = nif.shape[3]
        else:
            n_volumes = 1

        if nif.ndim != 2:
            n_slices = nif.shape[2]
        else:
            n_slices = 1

        # Can be '2D'
        mr_acquisition_type = nif.get_json_info('MRAcquisitionType')
        if mr_acquisition_type != '2D':
            # mr_acquisition_type is None or 3D
            raise NotImplementedError("MR acquisition type is not 2D.")

        # Offset time in ms to add a buffer to the imprecise timing of the DICOMS
        offset = 5000
        acq_start_time = int(nif.get_acquisition_start_time()) - offset
        acq_stop_time = int(nif.get_acquisition_stop_time()) + offset

        trigger_times = self.get_trigger_times(acq_start_time, acq_stop_time)
        if len(trigger_times) == 0:
            raise ValueError("No trigger times found in the specified range.")

        fat_suppression = nif.get_fat_sat_option()

        # If fat suppression, discard half the triggers
        if fat_suppression:
            # If there is fat sat, the 2nd, 4th, ... times correspond to a fat sat trigger
            # We only keep the imaging triggers (i.e.: 1st, 3rd, ... -> index 0, 2, 4...)
            trigger_times = trigger_times[::2]

        # Make sure we have the expected number of triggers
        if len(trigger_times) != n_volumes * n_slices:
            raise ValueError("Not enough trigger times for the specified number of volumes and slices.")

        slice_timing_start = nif.get_slice_timing()

        repetition_slice_excitation = nif.get_excitation_tr()

        # If the slice timing is lower than the TR excitation, this is an interleaved multi-slice acquisition
        # (more than one slice acquired within a TR excitation)
        # Remove slices that are at 0 ms (acquired during the first TR excitation)
        # If there are other slices acquired before the next TR excitation, then this is interleaved
        # multi-slice
        if np.any(slice_timing_start[slice_timing_start > 0] < repetition_slice_excitation):
            raise NotImplementedError("Interleaved multi-slice acquisition detected, but not implemented.")

        deltat_slice = nif.get_slice_tr()

        # Order according to slice timing
        slice_timing_from_triggers = np.zeros(n_volumes * n_slices, dtype=float)
        slice_timing_order = np.argsort(slice_timing_start)
        for i_vol in range(n_volumes):
            slice_timing_from_triggers[i_vol * n_slices:(i_vol + 1) * n_slices] = trigger_times[i_vol * n_slices:(i_vol + 1) * n_slices][slice_timing_order]

        # Get when the middle of k-space was acquired
        fourier = nif.get_partial_fourier()

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


class PmuRespLog(PmuResp):
    """ Read Siemens PMU log file (.log extension) located in %simmeasdata%\\Physio_Logfiles """

    def __init__(self, fname_log: str, fname_triggers=None, time_offset=0):
        data = self.read_pmu(fname_log)

        self.fname = fname_log
        if data.get('Time_tics') is None:
            raise ValueError("The log file does not contain the required 'Time_tics' field.")
        if data.get('RESP') is None:
            raise ValueError("The log file does not contain the required 'RESP' field.")

        self._data = np.array(data.get('RESP'))
        # A time tic corresponds to 2.5ms, convert to ms
        self._start_time_mdh = min(data.get('Time_tics')) * 2.5
        self._stop_time_mdh = max(data.get('Time_tics')) * 2.5
        self.start_time_mpcu = None
        self.stop_time_mpcu = None

        self.time_offset = 0
        self.adjust_start_time(time_offset)

        self.data_triggers = None
        if fname_triggers is not None:
            if fname_triggers is not None:
                raise NotImplementedError("Triggers using external file are not supported for PMU log files yet.")
        else:
            self.data_triggers = self._add_triggers_in_data()

        self.min = 0
        self.max = 4095

    def read_pmu(self, fname_log: str):
        return read_pmu_log_file(fname_log)

    def _add_triggers_in_data(self):
        data_with_triggers = []
        time_since_last_trig = 0
        data = self.get_data()
        data_mean = np.mean(data)
        time_between_datapoints = (self.timepoints[-1] - self.timepoints[0]) / len(self.timepoints)

        # Add triggers (5000) in the data when a new resp cycle starts
        # 0 1000 2000 3000 4000 3000 2000 1000 0  ->     0 1000 2000 5000 3000 4000 3000 2000 1000 0
        prev_data = data[0]
        for i_data, a_data in enumerate(data):

            if time_since_last_trig > 500 and prev_data <= data_mean < a_data:
                data_with_triggers.append(5000)
                time_since_last_trig = 0

            data_with_triggers.append(a_data)
            prev_data = a_data
            time_since_last_trig += time_between_datapoints

        return np.array(data_with_triggers)


class PmuExtLog(PmuExt):
    """ Read Siemens PMU ext1 log file (.log extension) located in %simmeasdata%\\Physio_Logfiles """

    def __init__(self, fname_log: str):
        data = self.read_pmu(fname_log)

        self.fname = fname_log
        if data.get('Time_tics') is None:
            raise ValueError("The log file does not contain the required 'Time_tics' field.")
        if data.get('EXT1') is None:
            raise ValueError("The log file does not contain the required 'EXT1' field.")

        self._data = np.array(data.get('EXT1'))
        # A time tic corresponds to 2.5ms, convert to ms
        self._start_time_mdh = min(data.get('Time_tics')) * 2.5
        self._stop_time_mdh = max(data.get('Time_tics')) * 2.5
        self.start_time_mpcu = None
        self.stop_time_mpcu = None

        self.data_triggers = self._add_triggers_in_data()

        self.time_offset = 0
        self.min = 0
        self.max = 1
        self.timepoints = self.get_all_times()

    def read_pmu(self, fname_log: str):
        return read_pmu_log_file(fname_log)

    def _add_triggers_in_data(self):
        data_with_triggers = []
        is_trig = False
        # Biopac seems to trig on both upwards and downwards transitions, we only want upwards
        going_up = True

        # Add triggers (5000) between the "1s" of the original data so that it looks like a PMU file
        # 0 0 0 0 1 1 1 1 0 0 1 1 1     ->     0 0 0 0 1 5000 1 1 1 0 0 1 5000 1 1
        for i_data, a_data in enumerate(self.get_data()):

            if is_trig:
                if a_data < 1:
                    is_trig = False
            else:
                if a_data > 0:
                    is_trig = True
                    data_with_triggers.append(a_data)
                    if going_up:
                        data_with_triggers.append(5000)
                    going_up = not going_up
                    continue

            data_with_triggers.append(a_data)

        # If the last data point was a trigger, add a final trigger since it was not added in the for loop
        if is_trig:
            data_with_triggers.append(5000)

        return np.array(data_with_triggers)


class PmuExtBiopac(PmuExtLog):
    """ Read Biopac and extract trigger data """

    def __init__(self, fname_biopac: str, nif: NiftiFile=None):
        data = self.read_pmu(fname_biopac)

        self.fname = fname_biopac
        if data.get('time') is None:
            raise ValueError("The biopac file does not contain the required 'time' field.")
        if data.get('trigs') is None:
            raise ValueError("The log file does not contain the required 'trigs' field.")

        # Trigger data is from 0 to 5, we convert it to 0 and 1
        # Decimate data by 10 (1 value every ms)
        self._data = (np.array(data.get('trigs'))[::10] > 4).astype(int)
        self._start_time_mdh = min(data.get('time')) * 60000
        self._stop_time_mdh = max(data.get('time')) * 60000
        self.start_time_mpcu = None
        self.stop_time_mpcu = None

        self.data_triggers = self._add_triggers_in_data()

        self.min = 0
        self.max = 1
        self.timepoints = self.get_all_times()

        if nif is not None:
            # Adjust start time so that it corresponds with the start of the acquisition
            trigger_times = self.get_trigger_times()
            if len(trigger_times) == 0:
                raise ValueError("No trigger times found in the PMU Ext file.")

            self._start_time_mdh = min(data.get('time')) * 60000 - trigger_times[0] + nif.get_acquisition_start_time()
            self._stop_time_mdh = max(data.get('time')) * 60000 - trigger_times[0] + nif.get_acquisition_start_time()

            self.timepoints = self.get_all_times()

    def read_pmu(self, fname_pmu):
        return read_biopac(fname_pmu)


class PmuRespBiopac(PmuResp):
    """ Read Biopac and extrac respiratory data """

    def __init__(self, fname_biopac: str, pmu_ext_biopac: PmuExtBiopac=None, time_offset=0):
        data = self.read_pmu(fname_biopac)

        self.fname = fname_biopac
        if data.get('time') is None:
            raise ValueError("The biopac file does not contain the required 'time' field.")
        if data.get('resp') is None:
            raise ValueError("The log file does not contain the required 'resp' field.")

        # Decimate data by 10 (1 value every ms)
        self._data = np.array(data.get('resp'))[::10]

        if pmu_ext_biopac is None:
            # If no trigger file is provided, we can't compute the start and stop times based on the triggers.
            self._start_time_mdh = min(data.get('time')) * 60000
            self._stop_time_mdh = max(data.get('time')) * 60000
        else:
            # This is so that the resp trace and the Ext trace are aligned in time
            if self.fname != pmu_ext_biopac.fname:
                raise ValueError("The Biopac file and the PMU Ext file must be the same.")
            trigger_times = pmu_ext_biopac.get_trigger_times()
            if len(trigger_times) == 0:
                raise ValueError("No trigger times found in the PMU Ext file.")
            self._start_time_mdh, self._stop_time_mdh = pmu_ext_biopac.get_start_and_stop_times()

        self.start_time_mpcu = None
        self.stop_time_mpcu = None

        self.time_offset = 0
        self.adjust_start_time(time_offset)

        self.data_triggers = None
        self.data_triggers = self._add_triggers_in_data()

        self.min = 0
        self.max = 1

    def read_pmu(self, fname_pmu):
        return read_biopac(fname_pmu)

    def _add_triggers_in_data(self):
        data_with_triggers = []
        time_since_last_trig = 0
        data = self.get_data()
        time_between_datapoints = (self.timepoints[-1] - self.timepoints[0]) / (len(self.timepoints) - 1)

        # Add triggers (5000) in the data when a new resp cycle starts
        # 0 1000 2000 3000 4000 3000 2000 1000 0  ->     0 1000 2000 5000 3000 4000 3000 2000 1000 0
        prev_data = data[0]
        for i_data, a_data in enumerate(data):
            # 20s span in number of indexes
            idx_span = int(20000 / time_between_datapoints)
            mean_start_ix = max(0, i_data - idx_span)
            if mean_start_ix == 0:
                data_mean = np.mean(data[mean_start_ix:i_data])
            else:
                # Faster running average
                data_mean += a_data / idx_span
                data_mean -= data[mean_start_ix - 1] / idx_span

            if time_since_last_trig > 500 and prev_data <= data_mean < a_data:
                data_with_triggers.append(5000)
                time_since_last_trig = 0

            data_with_triggers.append(a_data)
            prev_data = a_data
            time_since_last_trig += time_between_datapoints

        return np.array(data_with_triggers)


def read_pmu_log_file(fname_log: str):
    """
    Reads a Siemens log file and returns the data as a numpy array.

    Args:
        fname_log (str): Filename of the Siemens log file

    Returns:
        numpy.ndarray: Data read from the log file
    """

    if os.path.splitext(fname_log)[1] != '.log':
        raise ValueError("The file must have a .log extension.")

    with open(fname_log, 'r') as f:
        lines = f.readlines()

    data = {}
    header = None
    for i_line, line in enumerate(lines):
        # Skip header
        if i_line == 0:
            header = line.strip('\n').split(' ')
            continue
        for i_value, value in enumerate(line.strip('\n').split(' ')):
            if header[i_value] not in data:
                data[header[i_value]] = []
            data[header[i_value]].append(int(value))

    return data


def read_pmu_dicom(fname_dicom):
    # raise NotImplementedError("Reading PMU data from DICOM files is not implemented yet.")
    ds = dcmread(fname_dicom)

    if 'PMUDATA' not in ds['ImageType'].value:
        raise ValueError("The DICOM file does not contain PMU data.")

    # Data is compressed with gzip and stored in the '(7FE1,1010)' tag
    # It can be padded with '\xb7' that needs to be removed if present
    pmu_compressed = ds['0x7FE11010'].value
    if pmu_compressed[-1] == 0xb7:
        # Remove the padding byte
        pmu_compressed = pmu_compressed[:-1]

    # decompressed is a string of a html file
    decompressed = gzip.decompress(pmu_compressed).decode()
    # I verified a dataset, it contains exactly the same data as the resp log file.
    # There is also information about slice timing


def read_biopac(fname_biopac):
    with open(fname_biopac, 'r') as f:
        lines = f.readlines()

    # Todo: Remove header if it's there
    # if lines[0].strip()

    data = {
        'time': [],
        'trigs': [],
        'resp': []
    }
    for i_line in range(9, len(lines)):
        line = lines[i_line].strip().split('\t')
        if len(line) == 3:
            data['time'].append(float(line[0]))
            data['trigs'].append(float(line[1]))
            data['resp'].append(float(line[2]))
        elif len(line) == 2:
            data['trigs'].append(float(line[0]))
            data['resp'].append(float(line[1]))
        else:
            NotImplementedError("2 or 3 channels are supported for Biopac")

    return data


class OutOfRangeError(Exception):
    """
    Exception raised when the requested time is out of range of the PMU data.
    """
