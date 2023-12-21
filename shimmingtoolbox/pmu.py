#!/usr/bin/python3
# -*- coding: utf-8 -*
#
# Read Siemens Physiological Log files
# Adapted from https://gist.github.com/rtrhd/6172344
#

import numpy as np
from scipy import signal

class PmuResp(object):
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
    def __init__(self, fname_pmu):
        """

        Args:
            fname_pmu (str): Filename of the Siemens .resp file
        """

        attributes = self.read_resp(fname_pmu)

        self.fname = attributes['fname']
        self.data = attributes['data']
        self.start_time_mdh = attributes['start_time_mdh']
        self.stop_time_mdh = attributes['stop_time_mdh']
        self.start_time_mpcu = attributes['start_time_mpcu']
        self.stop_time_mpcu = attributes['stop_time_mpcu']
        self.max = attributes['max']
        self.min = attributes['min']

    def read_resp(self, fname_pmu):
        """
        Read a Siemens Physiological Log file. Returns a tuple with the logging data as numpy integer array and times
        in the form of milliseconds past midnight.

        Args:
            fname_pmu: Filename of the Siemens .resp file

        Returns:
            dict: A dict containing the ``fname_pmu`` infos. Contains the following keys:

                  * ``fname``
                  * ``data``
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

        # Drop the '6003\r\n' at the end
        clean_fields = clean_fields[:-1]

        # Returned values will be a numpy array of ints
        data = np.asarray([int(field) for field in clean_fields])
        length = len(data)
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

        # Define the filter parameters
        fs = 1 // ((stop_time_mdh - start_time_mdh) / 1000 / (length - 1))
        cutoff_freq = 0.4  # Cutoff frequency (Hz)
        nyquist_freq = 0.5 * fs  # Nyquist frequency
        order = 4  # Filter order

        # Create a low-pass Butterworth filter
        b, a = signal.butter(order, cutoff_freq / nyquist_freq, btype='low')
        data_cleaned = signal.filtfilt(b, a, data_cleaned)

        attributes = {
            'fname': fname_pmu,
            'data': data_cleaned,
            'start_time_mdh': start_time_mdh,
            'stop_time_mdh': stop_time_mdh,
            'start_time_mpcu': start_time_mpcu,
            'stop_time_mpcu': stop_time_mpcu,
            'max': 4095,
            'min': 0
        }

        return attributes

    def get_times(self, start_time=None, stop_time=None):
        """
        Get the times in ms at which the respiration took place.

            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            np.ndarray: Array containing the timepoints in ms of each data
        """
        times = self._get_all_times()
        start_idx, stop_idx = self._get_time_indexes(start_time, stop_time)

        # +1 is needed to include the stop_idx
        return times[start_idx:stop_idx + 1]

    def _get_all_times(self):
        raster = float(self.stop_time_mdh - self.start_time_mdh) / (len(self.data) - 1)
        times = (self.start_time_mdh + raster * np.arange(len(self.data)))  # ms
        return times

    def _get_time_indexes(self, start_time=None, stop_time=None):
        times = self._get_all_times()
        if start_time is None:
            start_time = times[0]
        if stop_time is None:
            stop_time = times[-1]

        start_idx = np.argmin(np.abs(times - start_time))
        stop_idx = np.argmin(np.abs(times - stop_time))

        return start_idx, stop_idx

    def get_resp_trace(self, start_time=None, stop_time=None):
        """
        Returns the resp trace between ``start_time`` and ``stop_time``

        Args:
            start_time (int): Start time in milliseconds past midnight
            stop_time (int): Stop time in milliseconds past midnight

        Returns:
            numpy.ndarray: Array with the resp trace between ``start_time`` and ``stop_time``
        """

        start_idx, stop_idx = self._get_time_indexes(start_time, stop_time)
        # +1 is needed to include the stop_idx
        return self.data[start_idx:stop_idx + 1]

    def interp_resp_trace(self, acquisition_times):
        """
        Interpolates ``data`` to the specified ``acquisition_times``

        Args:
            acquisition_times (numpy.ndarray): Array of the times in milliseconds past midnight of the desired
                                               times to interpolate the resp_trace. Times must be within
                                               ``self.start_time_mdh`` and ``self.stop_time_mdh``

        Returns:
            numpy.ndarray: Array with interpolated times with the same shape as ``acquisition_times``
        """
        if np.any(self.start_time_mdh > acquisition_times) or np.any(self.stop_time_mdh < acquisition_times):
            raise RuntimeError("acquisition_times don't fit within time limits for resp trace")

        times = self.get_times()
        interp_data = np.interp(acquisition_times, times, self.data)

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

        pressures = self.get_resp_trace(start_time=start_time, stop_time=stop_time)

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

        pressures = self.get_resp_trace(start_time=start_time, stop_time=stop_time)
        mean_p = self.mean(start_time=start_time, stop_time=stop_time)

        return np.sqrt(np.mean((pressures - mean_p)**2))
