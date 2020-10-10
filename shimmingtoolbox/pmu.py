#!/usr/bin/python3
# -*- coding: utf-8 -*
#
# Low level read of Siemens Physiological Log files
# Adapted from https://gist.github.com/rtrhd/6172344
#

import numpy as np


def read_resp(fname_pmu):
    """
    Read a Siemens Physiological Log file. Returns a tuple
    with the logging data as numpy integer array and times
    in the form of milliseconds past midnight.

    Args:
        fname_pmu: srt or file object
                   a file name to open or an already opened file-like object
    Returns:
        tuple:
            data_trimmed: centered around 0 (goes from -2047 to 2048)
            start_time_mdh
            stop_time_mdh
            start_time_mpcu
            stop_time_mpcu
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
    data_cleaned = data[data < 4096] - 2048

    return data_cleaned, start_time_mdh, stop_time_mdh, start_time_mpcu, stop_time_mpcu