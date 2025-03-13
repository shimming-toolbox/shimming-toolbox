#!/usr/bin/python3
# -*- coding: utf-8 -*
#
# Read Siemens Physiological Log files
# Adapted from https://gist.github.com/rtrhd/6172344
#

import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    def __init__(self, fname_pmu: str, time_offset=0):
        """

        Args:
            fname_pmu (str): Filename of the Siemens .resp file
            time_offset (int): Time offset in ms to what is read in the .resp file
        """

        attributes = self.read_resp(fname_pmu)

        self.fname = attributes['fname']
        self.data = attributes['data']
        self.start_time_mdh = attributes['start_time_mdh']
        self.stop_time_mdh = attributes['stop_time_mdh']
        self.start_time_mpcu = attributes['start_time_mpcu']
        self.stop_time_mpcu = attributes['stop_time_mpcu']
        self.data_triggers = attributes['data_triggers']
        self.max = attributes['max']
        self.min = attributes['min']
        self.time_offset = 0
        self.adjust_start_time(time_offset)
        self.timepoints = self.get_all_times()

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

        # Drop the '6003\r\n' at the end
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

        # start_time_mdh += 1070
        # stop_time_mdh += 1070
        # start_time_mpcu -= 8370
        # stop_time_mpcu -= 8370

        # GRE
        # start_time_mdh += 1000
        # stop_time_mdh += 1000

        attributes = {
            'fname': fname_pmu,
            'data': data_cleaned,
            'data_triggers': data,
            'start_time_mdh': start_time_mdh,
            'stop_time_mdh': stop_time_mdh,
            'start_time_mpcu': start_time_mpcu,
            'stop_time_mpcu': stop_time_mpcu,
            'max': 4095,
            'min': 0
        }

        return attributes

    def adjust_start_time(self, time_offset: int):
        """
        Offset the start and end time of the PMU data
        Args:
            time_offset (int): Time offset in ms to what is read in the .resp file

        """
        old_offset = self.time_offset
        self.start_time_mdh += time_offset - old_offset
        self.stop_time_mdh += time_offset - old_offset
        self.start_time_mpcu += time_offset - old_offset
        self.stop_time_mpcu += time_offset - old_offset
        self.time_offset = time_offset

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

    def get_all_times(self):
        """
        Get all the timepoints from the respiratory file (in ms).

        Returns:
            np.ndarray: Array containing the timepoints in ms of each data
        """
        raster = float(self.stop_time_mdh - self.start_time_mdh) / (len(self.data) - 1)
        times = (self.start_time_mdh + raster * np.arange(len(self.data)))  # ms
        return times

    def _get_time_indexes(self, start_time=None, stop_time=None):
        """
        Get the indexes of the data corresponding to the start and stop times

        Args:
            start_time:
            stop_time:

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
            # TODO: Explore why pmulog sequence raises an error for pmulog sequence
            logger.warning("acquisition_times don't fit within time limits for resp trace")
            start_offset = np.min(acquisition_times - self.start_time_mdh)
            stop_offset = np.min(self.stop_time_mdh - acquisition_times)
            logger.debug(f"start_offset: {start_offset}")
            logger.debug(f"stop_offset: {stop_offset}")
            # raise RuntimeError("acquisition_times don't fit within time limits for resp trace")

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

    def get_trigger_times(self, start_time=None, stop_time=None):
        """
        Returns the trigger times in ms of the resp trace. These triggers estimate the beginning of a new respiratory
        cycle

        Returns:
            numpy.ndarray: Array with the trigger times in ms of the resp trace
        """
        index_start, index_stop = self._get_time_indexes(start_time, stop_time)

        trigger_times = []
        i = 0
        for data in self.data_triggers:
            if data > 4096:
                if data != 5000:
                    logger.warning(f"Trigger value {data} not recognized")
                if index_start <= i <= index_stop:
                    trigger_times.append(float(self.timepoints[i]))
            else:
                i += 1

        logger.debug(f"Trigger times: {trigger_times}")

        return trigger_times

    def get_mean_trigger_span(self):
        """
        Returns the mean time between triggers in ms

        Returns:
            float: Mean time between triggers in ms
        """
        trigger_times = self.get_trigger_times()
        trigger_span = np.diff(trigger_times)
        return np.mean(trigger_span)
