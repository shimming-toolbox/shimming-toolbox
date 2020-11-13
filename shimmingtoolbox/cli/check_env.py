#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check the installation of the dependencies

import click
import subprocess
from typing import Dict, Tuple, List

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Verify that the dependencies are installed and available to the toolbox."
)
def check_dependencies():
    """Verifies dependencies are installed by calling helper functions and
    formatting output accordingly.
    """
    # 0 indicates prelude is installed.
    prelude_exit_code: int = check_prelude_installation()
    # negating condition because 0 indicates prelude is installed.
    if not prelude_exit_code:
        print(get_prelude_version())
    else:
        print(f"Error {prelude_exit_code}: prelude is not installed or not in your PATH.")

    # 0 indicates dcm2niix is installed.
    dcm2niix_exit_code: int = check_dcm2niix_installation()
    # negating condition because 0 indicates dcm2niix is installed.
    if not dcm2niix_exit_code:
        print(get_dcm2niix_version())
    else:
        print(f"Error {dcm2niix_exit_code}: dcm2niix is not installed or not in your PATH.")

    return


def check_prelude_installation() -> int:
    """Checks that ``prelude`` is installed.

    This function calls ``which prelude`` and checks the exit code to verify
    that ``prelude`` is installed.

    Returns:
        int: Exit code. 0 on success, nonzero on failure.
    """

    return subprocess.check_call(['which', 'prelude'])


def check_dcm2niix_installation() -> int:
    """Checks that ``dcm2niix`` is installed.

    This function calls ``which dcm2niix`` and checks the exit code to verify
    that ``dcm2niix`` is installed.

    Returns:
        int: Exit code. 0 on success, nonzero on failure.
    """
    return subprocess.check_call(['which', 'dcm2niix'])


def get_prelude_version() -> str:
    """Gets the ``prelude`` installation version.

    This function calls ``prelude --help`` and parses the output to obtain the
    installation version.

    Returns:
        str: Version of the ``prelude`` installation.
    """
    # `prelude --help` returns an error code and output is in stderr
    prelude_help = subprocess.run(["prelude", "--help"], capture_output=True, encoding="utf-8")
    # If the behaviour of FSL prelude changes to output help in stdout with a
    # 0 exit code, this function must fail loudly so we can update its
    # behaviour accordingly:
    assert prelude_help.returncode != 0
    # we're capturing stderr instead of stdout
    help_output: str = prelude_help.stderr.rstrip()
    # remove beginning newline and drop help info to keep version info
    version: str = help_output.split("\n\n")[0].replace("\n","", 1)
    return version


def get_dcm2niix_version() -> str:
    """Gets the ``dcm2niix`` installation version.

    This function calls ``dcm2niix --version`` and captures the output to
    obtain the installation version.

    Returns:
        str: Version of the ``dcm2niix`` installation.
    """
    # `dcm2niix --version` returns an error code and output is in stderr
    dcm2niix_version: str = subprocess.run(["dcm2niix", "--version"], capture_output=True, encoding="utf-8")
    # If the behaviour of dcm2niix changes to output help with a 0 exit code,
    # this function must fail loudly so we can update its behaviour
    # accordingly:
    assert dcm2niix_version.returncode != 0
    version_output: str = dcm2niix_version.stdout.rstrip()
    return version_output

