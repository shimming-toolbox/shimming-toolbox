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

from shimmingtoolbox import __version__, __dir_repo__
from shimmingtoolbox.utils import check_exe

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

    # Shimming Toolbox
    print(f"Shimming Toolbox version: {__version__}")

    # Plugin
    plugin_version = get_plugin_version()
    print(f"Plugin version: {plugin_version}\n")

    # Git
    print(f"Git version: {get_git_version()}")

    # Prelude
    prelude_check_msg = check_name.format("prelude")
    print_line(prelude_check_msg)
    check_prelude_installation()

    # # dcm2niix
    # dcm2niix now comes bundled with shimming toolbox. Therefore we don't need to check if it is in the path since it
    # is already in the environment
    # dcm2niix_check_msg = check_name.format("dcm2niix")
    # print_line(dcm2niix_check_msg)
    # check_dcm2niix_installation()

    # SCT
    sct_check_msg = check_name.format("Spinal Cord Toolbox")
    print_line(sct_check_msg)
    check_sct_installation()

    return


def check_prelude_installation():
    """Checks that ``prelude`` is installed.

    This function calls ``which prelude`` and checks the exit code to verify that ``prelude`` is installed.

    Returns:
        bool: True if prelude is installed, False if not.
    """
    try:
        if sys.platform == 'win32':
            subprocess.check_call(['where', 'prelude'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.check_call(['which', 'prelude'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as error:
        print_fail()
        print(f"Error {error.returncode}: prelude is not installed or not in your PATH.")
        return False
    else:
        print_ok()
        print("    " + get_prelude_version().replace("\n", "\n    "))
        return True


def check_dcm2niix_installation():
    """Checks that ``dcm2niix`` is installed.

    This function calls ``which dcm2niix`` and checks the exit code to verify that ``dcm2niix`` is installed.

    Returns:
        bool: True if dcm2niix is installed, False if not.
    """
    try:
        if sys.platform == 'win32':
            subprocess.check_call(['where', 'dcm2niix'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.check_call(['which', 'dcm2niix'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(error)
        print_fail()
        print(f"Error {error.returncode}: dcm2niix is not installed or not in your PATH.")
        return False
    else:
        print_ok()
        print("    " + get_dcm2niix_version().replace("\n", "\n    "))
        return True


def check_sct_installation():
    """Checks that ``SCT`` is installed.

    This function calls ``which sct_check_dependencies`` and checks the exit code to verify that ``sct`` is installed.

    Returns:
        bool: True if sct is installed, False if not.
    """
    try:
        subprocess.check_call(['sct_check_dependencies', '-short'], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, shell=True)
    except subprocess.CalledProcessError as error:
        print_fail()
        print(f"Error {error.returncode}: Spinal Cord Toolbox is not installed or not in your PATH.")
        return False
    else:
        print_ok()
        print("    " + get_sct_version().replace("\n", "\n    "))
        return True


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
    version: str = help_output.split("\n\n")[0].replace("\n", "", 1)
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


def get_sct_version() -> str:
    """Gets the ``sct`` installation version.

    This function calls ``sct_check_dependencies -short`` and captures the output to
    obtain the installation version.

    Returns:
        str: Version of the ``SCT`` installation.
    """
    # `sct_check_dependencies -short` returns
    sct_version: str = subprocess.run(["sct_version", "-short"], capture_output=True, encoding="utf-8")
    if sct_version.returncode != 0:
        raise subprocess.CalledProcessError("Error while getting SCT's version")
    version_output: str = sct_version.stdout.rstrip()

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
    git_version = get_git_version()
    plugin_version = get_plugin_version()

    print(f"Shimming Toolbox version:\n{__version__}\n\nPlugin version:\n{plugin_version}\n\nGit information:\n{git_version}\n\n"
          f"Environment info:\n{env_info}")
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
                f"{python_implementation} {python_full_version}\n" +
                f"{cpu_usage}\n" +
                f"{ram_usage}"
                )
    return env_info


def get_git_version():
    """ Get the git version of the repository
    Returns:
        string: branch-commit
    """
    st_commit = None
    st_branch = None
    if check_exe("git") and os.path.isdir(os.path.join(__dir_repo__, ".git")):
        st_commit = _get_commit() or st_commit
        st_branch = _get_branch() or st_branch

    if st_commit is None:
        return "Not a Git repository"

    return f"{st_branch}-{st_commit}"


def _get_branch():
    """
    """

    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__dir_repo__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def _get_commit(path_to_git_folder=None):
    """
    """
    if path_to_git_folder is None:
        path_to_git_folder = __dir_repo__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def get_plugin_version():
    """ Get the version of the plugin
    Returns:
        string: version
    """
    try:
        from fsleyes_plugin_shimming_toolbox import __version__ as plugin_version
        return plugin_version
    except ImportError:
        return "Not installed"
