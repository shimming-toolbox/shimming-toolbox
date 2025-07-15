#!/usr/bin/env python
# -*- coding: utf-8
# Misc functions

import numpy as np
import os
import tqdm
import subprocess
import logging
import nibabel as nib
import json
import time
import functools
from scipy import ndimage as nd
import hashlib
import shutil

logger = logging.getLogger(__name__)


def run_subprocess(cmd):
    """Wrapper for ``subprocess.run()``.

    Args:
        cmd (list): list of arguments to be passed to the command line
    """
    logger.debug(f"Command to run on the terminal:\n{' '.join(cmd)}")
    try:

        subprocess.run(
            cmd,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        msg = "Return code: ", err.returncode, "\nOutput: ", err.stderr
        print(msg)
        raise err


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


def ms_past_midnight_to_iso_time(ms_past_midnight):
    """
    Convert milliseconds past midnight to a formatted string.

    Args:
        ms_past_midnight (int): Time past midnight in milliseconds.

    Returns:
        str: Time formatted as "HHMMSS.mmmmmm"
    """
    hours = int(ms_past_midnight / 1000 / 3600)
    minutes = int((ms_past_midnight - (hours * 1000 * 3600)) / 1000 / 60)
    secs = int((ms_past_midnight - (hours * 1000 * 3600) - (minutes * 1000 * 60)) / 1000)
    us = int(1000 * (ms_past_midnight - (hours * 1000 * 3600) - (minutes * 1000 * 60) - (secs * 1000)))

    # Error check
    if 0 > hours >= 24 or 0 > minutes >= 60 or 0 > secs >= 60 or 0 > us >= 1000000:
        raise ValueError("Input ms_past_midnight is not a valid time past midnight in milliseconds")

    return f"{hours:02}{minutes:02}{secs:02}.{us:06}"


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


def set_all_loggers(verbose, list_exclude=('matplotlib', 'indexed_gzip')):
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


def montage(X):
    """Concatenates images stored in a 3D array
    Args:
        X (numpy.ndarray): 3D array with the last dimension being the one in which the images are concatenated.
    Returns:
        numpy.ndarray: 2D array of concatenated images.
    """
    X = np.rot90(X)
    x, y, n_images = np.shape(X)
    mm = np.floor(np.sqrt(n_images)).astype(int)
    nn = np.ceil(n_images/mm).astype(int)
    result = np.empty((mm * x, nn * y))
    result.fill(np.nan)
    image_id = 0
    for k in range(mm):
        for j in range(nn):
            if image_id >= n_images:
                break
            slice_m, slice_n = k * x, j * y
            result[slice_m:slice_m + x, slice_n:slice_n + y] = X[:, :, image_id]
            image_id += 1
    return result


def save_nii_json(nii, json_data, fname_output):
    """ Save the nii to a nifti file and dict to a json file.

    Args:
        nii (nib.Nifti1Image): Nibabel object containing data save.
        json_data (dict): Dictionary containing the json sidecar associated with the nibabel object.
        fname_output (str): Output filename, supported types : '.nii', '.nii.gz'
    """
    # Make sure output filename is valid
    if fname_output[-4:] != '.nii' and fname_output[-7:] != '.nii.gz':
        raise ValueError(f"Output filename: {fname_output} must have one of the following extensions: '.nii', "
                         "'.nii.gz'")

    # Create output directory if it does not exist
    create_output_dir(fname_output, is_file=True)

    # Save NIFTI
    nib.save(nii, fname_output)

    # Save json
    fname_json = fname_output.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)


def timeit(func):
    """ Decorator to time a function. Decorate a function: @timeit on top of the function definition. The elapsed time
    will output in debug mode
    """

    @functools.wraps(func)
    def timed(*args, **kw):

        ts = time.time()
        # Call the original function
        result = func(*args, **kw)
        te = time.time()

        # Log the output
        logger.debug(f"Function: {func.__name__} took {te - ts:.4}s to run")

        return result

    return timed


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Args:
        data (numpy.ndarray)): array of any dimension
        invalid (numpy.ndarray): a binary array of same shape as 'data'. True cells set where data
                                 value should be replaced.
                                 If None (default), use: invalid  = np.isnan(data)

    Returns:
        numpy.ndarray: Return a filled array.
    """

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def are_niis_equal(nii1:nib.nifti1.Nifti1Image, nii2:nib.nifti1.Nifti1Image):
    return hashlib.sha256(nii1.get_fdata().tobytes()).hexdigest() == \
           hashlib.sha256(nii2.get_fdata().tobytes()).hexdigest() and \
           hashlib.sha256(nii1.affine.tobytes()).hexdigest() == \
           hashlib.sha256(nii2.affine.tobytes()).hexdigest()


def are_jsons_equal(json1:dict, json2:dict):
    json1_bytes = json.dumps(json1).encode('utf-8')
    json2_bytes = json.dumps(json2).encode('utf-8')
    return hashlib.sha256(json1_bytes).hexdigest() == \
           hashlib.sha256(json2_bytes).hexdigest()


def check_exe(name):
    """
    Ensure that a program exists and can be executed
    """
    _, filename = os.path.split(name)
    # Case 1: Check full filepath directly (which may point to a location not on the PATH)
    if os.path.isfile(name) and os.access(name, os.X_OK):
        return True
    # Case 2: Check filename only via the PATH
    elif shutil.which(filename) and os.access(shutil.which(filename), os.X_OK):
        return True
    else:
        return False
