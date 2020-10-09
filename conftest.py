###############################################################################
#      Configure tests so that pytest downloads testing data every time.      #
###############################################################################
import sys
import os
import logging
from pathlib import Path
import subprocess

import pytest
from typing import Mapping
from hashlib import md5

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.cli.download_data import download_data

logger = logging.getLogger(__name__)


def test_data_path():
    # Get the directory where this current file is saved
    test_path = Path(__file__).resolve().parent

    # create tmp folder
    tmp_path = test_path / __dir_testing__
    if not tmp_path.exists():
        tmp_path.mkdir()
    return tmp_path


@pytest.fixture
def test_data_path_fixture():
    return test_data_path()


@pytest.fixture(params=[pytest.param(0, marks=pytest.mark.prelude)])
def test_prelude_installation():
    # note that subprocess.check_call() returns 0 on success, so it must be
    # negated for the assertion.
    assert not subprocess.check_call(['which', 'prelude'])
    return


@pytest.fixture(params=[pytest.param(0, marks=pytest.mark.dcm2niix)])
def test_dcm2niix_installation():
    # note that subprocess.check_call() returns 0 on success, so it must be
    # negated for the assertion.
    assert not subprocess.check_call(['which', 'dcm2niix'])
    return


def pytest_sessionstart():
    """Download shimmingtoolbox testing_data prior to test collection."""
    logger.info("Downloading shimmingtoolbox test data")
    test_data_location = test_data_path()
    try:
        download_data(
            [
                # '--verbose',
                '--output',
                test_data_location,
                'testing_data',
            ]
        )
    # click sends a SystemExit upon command completion that needs to be caught.
    except SystemExit:
        return


@pytest.fixture(scope="session", autouse=True)
def test_data_integrity(request):
    files_checksums = dict()
    for root, _, files in os.walk(test_data_path()):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            files_checksums[fname] = chksum
    request.addfinalizer(lambda: check_testing_data_integrity(files_checksums))


def checksum(fname: os.PathLike) -> str:
    with open(fname, 'rb') as f:
        data = f.read()
    return md5(data).hexdigest()


def check_testing_data_integrity(files_checksums: Mapping[os.PathLike, str]):
    changed = []
    new = []
    missing = []

    after = []

    test_data_location = test_data_path()

    for root, _, files in os.walk(test_data_location):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            after.append(fname)

            if fname not in files_checksums:
                logger.warning(
                    f"Discovered new file in testing_data that didn't exist before: {(fname, chksum)}"
                )
                new.append((fname, chksum))

            elif files_checksums[fname] != chksum:
                logger.error(
                    f"Checksum mismatch for test data: {fname}. Got {chksum} instead of {files_checksums[fname]}"
                )
                changed.append((fname, chksum))

    for fname, chksum in files_checksums.items():
        if fname not in after:
            logger.error(f"Test data missing after test:a: {fname}")
            missing.append((fname, chksum))

    assert not changed
    # assert not new
    assert not missing
