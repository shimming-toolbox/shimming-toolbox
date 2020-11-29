#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file provides 2 CLI entrypoints and associated helper functions:
#  - a tool to check dependency installation, availability in PATH, and version
#  - a tool to dump details about the environment and shimmingtoolbox
#    installation

import click
import subprocess
import os
import platform
import psutil
import sys

from typing import Dict, Tuple, List

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_ok(more=None):
    print("[{}OK{}]{}".format(bcolors.OKGREEN, bcolors.ENDC, more if more is not None else ""))


def print_warning(more=None):
    print("[{}WARNING{}]{}".format(bcolors.WARNING, bcolors.ENDC, more if more is not None else ""))


def print_fail(more=None):
    print("[{}FAIL{}]{}".format(bcolors.FAIL, bcolors.ENDC, more if more is not None else ""))


def print_line(string):
    """print without carriage return"""
    sys.stdout.write(string.ljust(52, '.'))
    sys.stdout.flush()


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Verify that the dependencies are installed and available to the toolbox."
)
def check_dependencies():
    """Verifies dependencies are installed by calling helper functions and
    formatting output accordingly.
    """
    check_name = "Check if {} is installed"

    # Prelude
    prelude_check_msg = check_name.format("prelude")
    print_line(prelude_check_msg)
    # 0 indicates prelude is installed.
    prelude_exit_code: int = check_prelude_installation()
    # negating condition because 0 indicates prelude is installed.
    if not prelude_exit_code:
        print_ok()
        print("    " + get_prelude_version().replace("\n", "\n    "))
    else:
        print_fail()
        print(f"Error {prelude_exit_code}: prelude is not installed or not in your PATH.")

    # dcm2niix
    dcm2niix_check_msg = check_name.format("dcm2niix")
    print_line(dcm2niix_check_msg)
    # 0 indicates dcm2niix is installed.
    dcm2niix_exit_code: int = check_dcm2niix_installation()
    # negating condition because 0 indicates dcm2niix is installed.
    if not dcm2niix_exit_code:
        print_ok()
        print("    " + get_dcm2niix_version().replace("\n", "\n    "))
    else:
        print_fail()
        print(f"Error {dcm2niix_exit_code}: dcm2niix is not installed or not in your PATH.")

    return


def check_prelude_installation() -> int:
    """Checks that ``prelude`` is installed.

    This function calls ``which prelude`` and checks the exit code to verify
    that ``prelude`` is installed.

    Returns:
        int: Exit code. 0 on success, nonzero on failure.
    """

    return subprocess.check_call(['which', 'prelude'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_dcm2niix_installation() -> int:
    """Checks that ``dcm2niix`` is installed.

    This function calls ``which dcm2niix`` and checks the exit code to verify
    that ``dcm2niix`` is installed.

    Returns:
        int: Exit code. 0 on success, nonzero on failure.
    """
    return subprocess.check_call(['which', 'dcm2niix'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Dumps environment and package details into stdout for debugging purposes."
)
def dump_env_info():
    """Dumps environment and package details into stdout for debugging purposes
    by calling helper functions to retrieve these details.
    """
    env_info = get_env_info()
    pkg_version = get_pkg_info()

    print(f"ENVIRONMENT INFO:\n{env_info}\n\nPACKAGE INFO:\n{pkg_version}")
    return


def get_env_info() -> str:
    """Gets information about the environment.

    This function gets information about the operating system, the host
    machine hardware, Python version & implementation, and Python location.

    Returns:
        str: A multiline string containing environment info.
    """

    os_name = os.name
    cpu_arch = platform.machine()
    platform_release = platform.release()
    platform_system = platform.system()
    platform_version = platform.version()
    python_full_version = platform.python_version()
    python_implementation = platform.python_implementation()

    cpu_usage = f"CPU cores: Available: {psutil.cpu_count()}, Used by ITK functions: {int(os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0))}"
    ram = psutil.virtual_memory()
    factor_MB = 1024 * 1024
    ram_usage = f'RAM: Total: {ram.total // factor_MB}MB, Used: {ram.used // factor_MB}MB, Available: {ram.available // factor_MB}MB'

    env_info = (f"{os_name} {cpu_arch}\n" +
                f"{platform_system} {platform_release}\n" +
                f"{platform_version}\n" +
                f"{python_implementation} {python_full_version}\n\n" +
                f"{cpu_usage}\n" +
                f"{ram_usage}"
                )
    return env_info


def get_pkg_info() -> str:
    """Gets package version.

    This function gets the version of shimming-toolbox.

    Returns:
        str: The version number of the shimming-toolbox installation.
    """
    import shimmingtoolbox as st
    pkg_version = st.__version__
    return pkg_version
