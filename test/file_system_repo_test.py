#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest
import tempfile
import errno

from shimmingtoolbox.language import English as notice
from shimmingtoolbox import __dir_testing__


# Repository tests for remove function

# Test Setup
def setup():
    with tempfile.TempDir() as test_directory:
        temp_file_name = os.path.join(test_directory.name, 'test_file_repo.txt')


# Happy path (File is removed)
@pytest.fixture(scope="session")
def test_remove_happy_path():
    setup()
    command_remove = remove( tempfile.gettempdir()+'/test_file_repo.txt' )
    
    assert command_remove.returncode == 0


# File to delete is not found
# Also checks default and parametrized error messages
@pytest.fixture(scope="session")
def test_remove_not_found():
    setup()
    command_remove = remove( tempfile.gettempdir()+'/nonsense.txt' )
    assert command_remove.returncode == 1
    
    test_error_message = 'You do not have that file'
    command_remove = remove( tempfile.gettempdir()+'/nonsense.txt', test_error_message )
    
    assert command_remove.returncode == 1  
    assert command_remove.value.args[0] == test_error_message
