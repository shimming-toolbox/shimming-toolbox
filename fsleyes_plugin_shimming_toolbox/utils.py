#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import subprocess

from pathlib import Path

logger = logging.getLogger(__name__)

HOME_DIR = str(Path.home())
PATH_ST_VENV = f"{HOME_DIR}/shimming-toolbox/python/envs/st_venv/bin"


def run_subprocess(cmd):
    """Wrapper for ``subprocess.run()`` that enables to input ``cmd`` as a full string
        (easier for debugging).

    Args:
        cmd (string): full command to be run on the command line
    """
    logging.info(f'{cmd}')
    try:
        env = os.environ.copy()
        # It seems to default to the Python executalble instead of the Shebang, removing it fixes it
        env["PYTHONEXECUTABLE"] = ""
        env["PATH"] = PATH_ST_VENV + ":" + env["PATH"]
        
        # stdout captures print, stderr captures logging and errors
        # Solution: Pipe stderr to stdout
        result = subprocess.run(
            cmd.split(' '),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            env=env
        )
        return result.stdout
    except subprocess.CalledProcessError as err:
        # Since std err was piped to stdout, we output stdout
        msg = "Return code: ", err.returncode, "\nOutput: ", err.stdout
        raise Exception(msg)
