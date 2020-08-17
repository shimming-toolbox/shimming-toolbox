#!/usr/bin/env python
# -*- coding: utf-8
# Misc functions

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


def st_progress_bar(*args, **kwargs):
    """Thin wrapper around `tqdm.tqdm` which checks `SCT_PROGRESS_BAR` muffling the progress
       bar if the user sets it to `no`, `off`, or `false` (case insensitive)."""
    do_pb = os.environ.get('SCT_PROGRESS_BAR', 'yes')
    if do_pb.lower() in ['off', 'no', 'false']:
        kwargs['disable'] = True

    return tqdm.tqdm(*args, **kwargs)
