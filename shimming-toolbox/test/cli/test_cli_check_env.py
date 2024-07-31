#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import re
from click.testing import CliRunner

import shimmingtoolbox.cli.check_env as st_ce


@pytest.mark.dcm2niix
@pytest.mark.prelude
@pytest.mark.sct
def test_check_dependencies(test_dcm2niix_installation, test_prelude_installation):
    runner = CliRunner()

    result = runner.invoke(st_ce.check_dependencies)
    assert result.exit_code == 0


def test_dump_env_info():
    runner = CliRunner()

    result = runner.invoke(st_ce.dump_env_info)
    assert result.exit_code == 0


def test_check_installation_errors():
    """Tests that the function returns False as expected
    """
    runner = CliRunner(env={'PATH': '/usr/bin'})

    result = runner.invoke(st_ce.check_dependencies, catch_exceptions=False)
    assert result.exit_code == 0
    assert result.stdout.count('FAIL') == 2


def test_check_prelude_installation():
    """Tests that the function returns an exit code as expected, does not test
    the exit code value itself, so it does not depend on prelude being
    installed.
    """
    check_prelude_installation_exit = st_ce.check_prelude_installation()
    assert isinstance(check_prelude_installation_exit, int)


def test_check_dcm2niix_installation():
    """Tests that the function returns an exit code as expected, does not test
    the exit code value itself, so it does not depend on dcm2niix being
    installed.
    """
    check_dcm2niix_installation_exit = st_ce.check_dcm2niix_installation()
    assert isinstance(check_dcm2niix_installation_exit, int)


def test_check_sct_installation():
    """Tests that the function returns a bool as expected. it does not depend on sct being
    installed.
    """
    check_sct_installation_exit = st_ce.check_sct_installation()
    assert isinstance(check_sct_installation_exit, bool)


@pytest.mark.prelude
def test_get_prelude_version(test_prelude_installation):
    """Checks prelude version output for expected structure.
    """
    prelude_version_info = st_ce.get_prelude_version()
    version_regex = r"Part of FSL.*\nprelude.*\nPhase.*\nCopyright.*"
    assert re.search(version_regex, prelude_version_info)


@pytest.mark.dcm2niix
def test_get_dcm2niix_version(test_dcm2niix_installation):
    """Checks dcm2niix version output for expected structure.
    """
    dcm2niix_version_info = st_ce.get_dcm2niix_version()
    version_regex = r"Chris.*\nv\d\.\d.\d{8}"
    assert re.search(version_regex, dcm2niix_version_info)


@pytest.mark.sct
def test_get_sct_version(test_sct_installation):
    """Checks sct version output for expected structure.
    """
    sct_version_info = st_ce.get_sct_version()
    version_regex = r"git*"
    assert re.search(version_regex, sct_version_info) or re.search(r"\d\.\d", sct_version_info)


def test_get_env_info():
    env_info = st_ce.get_env_info()
    assert isinstance(env_info, str)


def test_version():
    assert type(st_ce.get_git_version()) == str


def test_plugin_version():
    assert type(st_ce.get_plugin_version()) == str
