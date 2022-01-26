#!/usr/bin/env python
# -*- coding: utf-8
# Misc functions

import numpy as np
import os
import tqdm
import subprocess
import logging


def run_subprocess(cmd):
    """Wrapper for ``subprocess.run()`` that enables to input ``cmd`` as a full string (easier for debugging).

    Args:
        cmd (string): full command to be run on the command line
    """
    logging.debug(f'{cmd}')
    try:
        subprocess.run(
            cmd.split(' '),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as err:
        msg = "Return code: ", err.returncode, "\nOutput: ", err.stderr
        raise Exception(msg)


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

    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def splitext(fname):
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
        elif isinstance(a_time, str) and \
                (len(a_time.split(':')) == 3) and \
                (len(a_time.split(':')[2].split('.')) == 2):
            split_colon = a_time.split(':')
            hours = int(split_colon[0])
            minutes = int(split_colon[1])
            seconds = int(split_colon[2].split('.')[0])
            micros = int(split_colon[2].split('.')[1])
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


def create_output_dir(path_output, is_file=False, output_folder_name="output"):
    """Given a path, create the directory if it doesn't exist.

    Args:
        path_output (str): Full path to either a folder or a file.
        is_file (bool): True if the ``path_output`` is for a file, else False.
        output_folder_name (str): Name of sub-folder.
    """

    if is_file:
        path_output_folder = os.path.dirname(path_output)
    else:
        path_output_folder = path_output
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)


def create_fname_from_path(path, file_default):
    """Given a path, make sure it is not a directory, if it is add the default filename, if not, return the path

    Args:
        path (str): filename or path to add the `file_default` to.
        file_default (str): Name of the file + ext (example.nii.gz) to add to the path if the path is a directory.

    Returns:
        str: Absolute path of a file
    """

    is_dir = os.path.splitext(path)[-1] == ''

    if is_dir:
        fname = os.path.join(path, file_default)
    else:
        fname = path

    return os.path.abspath(fname)


def set_all_loggers(verbose, list_exclude=('matplotlib',)):
    """ Set all loggers in the root manager to the verbosity level. Exclude any logger with the name in list_exclude

    Args:
        verbose (str): Verbosity level: 'info', 'debug', 'warning', 'critical', 'error'
        list_exclude: List of string to exclude from logging
    """
    loggers = []
    # For every logger name
    for name in logging.root.manager.loggerDict:

        # Exclude the setting level if it is in the excluded list
        is_excluded = False
        for exclude in list_exclude:
            if name.startswith(exclude):
                is_excluded = True

        if not is_excluded:
            loggers.append(logging.getLogger(name))

    for a_logger in loggers:
        a_logger.setLevel(verbose.upper())


def montage(X, colormap='gray', title=None, vmin=None, vmax=None):
    """Concatenates images stored in a 3D array
    Args:
        X (numpy.ndarray): 3D array with the last dimension being the one in which the different images are stored
        colormap (str): Colors in which the montage will be displayed.
        title (str): Title to display above the figure.
        vmin (float): Minimum display range value. If None, set the the min value of X.
        vmax (float): Maximum display range value. If None, set the the max value of X.
    """
    X = np.rot90(X)
    x, y, n_images = np.shape(X)
    mm = np.floor(np.sqrt(n_images)).astype(int)
    nn = np.ceil(n_images/mm).astype(int)
    result = np.zeros((mm * x, nn * y))
    image_id = 0
    for k in range(mm):
        for j in range(nn):
            if image_id >= n_images:
                break
            slice_m, slice_n = k * x, j * y
            result[slice_m:slice_m + x, slice_n:slice_n + y] = X[:, :, image_id]
            image_id += 1
    return result
