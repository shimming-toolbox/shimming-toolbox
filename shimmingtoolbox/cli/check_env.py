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
        print("prelude is installed.")
    else:
        print(f"Error {prelude_exit_code}: prelude is not installed or not in your PATH.")

    # 0 indicates dcm2niix is installed.
    dcm2niix_exit_code: int = check_dcm2niix_installation()
    # negating condition because 0 indicates dcm2niix is installed.
    if not dcm2niix_exit_code:
        print("dcm2niix is installed.")
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
