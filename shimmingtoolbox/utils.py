#!/usr/bin/env python
# -*- coding: utf-8
# Misc functions

import numpy as np
import os
import tqdm
import subprocess
import logging


def run_subprocess(cmd):
    """
    Wrapper for ``subprocess.run()`` that enables to input ``cmd`` as a full string (easier for debugging).

    Args:
        cmd (string): full command to be run on the command line
    """
    logging.debug('{}'.format(cmd))
    subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, text=True, check=True)


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    Args:
        fname: absolute or relative file name. Example: ``t2.nii``
        suffix: suffix. Example: ``_mean``

    :Return: file name string with suffix. Example: ``t2_mean.nii``

    Examples:

    - ``add_suffix(t2.nii, _mean)`` -> ``t2_mean.nii``
    - ``add_suffix(t2.nii.gz, a)`` -> ``t2a.nii.gz``
    """

    def _splitext(fname):
        """
        Split a fname (folder/file + ext) into a folder/file and extension.

        Note: for .nii.gz the extension is understandably .nii.gz, not .gz
        (``os.path.splitext()`` would want to do the latter, hence the special case).
        """
        dir, filename = os.path.split(fname)
        for special_ext in ['.nii.gz', '.tar.gz']:
            if filename.endswith(special_ext):
                stem, ext = filename[:-len(special_ext)], special_ext
                return os.path.join(dir, stem), ext
        # If no special case, behaves like the regular splitext
        stem, ext = os.path.splitext(filename)
        return os.path.join(dir, stem), ext

    stem, ext = _splitext(fname)
    return os.path.join(stem + suffix + ext)


def iso_times_to_ms(iso_times):
    """
    Convert dicom acquisition times to ms

    Args:
        iso_times (numpy.ndarray): 1D array of time strings from dicoms.
                                   Suported formats: "HHMMSS.mmmmmm" or "HH:MM:SS.mmmmmm"

    Returns:
        numpy.ndarray: 1D array of times in milliseconds
    """

    ms_times = []

    for a_time in iso_times:
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


def st_progress_bar(*args, **kwargs):
    """Thin wrapper around `tqdm.tqdm` which checks `SCT_PROGRESS_BAR` muffling the progress
       bar if the user sets it to `no`, `off`, or `false` (case insensitive)."""
    do_pb = os.environ.get('SCT_PROGRESS_BAR', 'yes')
    if do_pb.lower() in ['off', 'no', 'false']:
        kwargs['disable'] = True

    return tqdm.tqdm(*args, **kwargs)
