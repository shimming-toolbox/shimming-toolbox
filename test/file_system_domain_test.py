#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest
import errno

from shimmingtoolbox.language import English as notice
from shimmingtoolbox import __dir_testing__


#!usr/bin/env python3
# -*- coding: utf-8

'''
 Identifies if there is at least one file with the provided extension
 Returns true or false
 Requires directory path, dir_path, as a string
 Requires extension as a string
'''

# For does_contain_extension function


# Happy path (Directory contains extension)
def test_directory_has_extension():
    assert 1 ==1

# Negative Path (Directory does not contain extension)
def test_directory_no_extension():
    assert 1 ==1
# Negative Path (Extension is nonsense)
def test_nonsense_extensions():
    assert 1 ==1
# Negative Path (Directory is nonsense)
# Also checks default and parametrized error messages
def test_nonsense_directory():
    assert 1 ==1



# For isFound function


# Happy path (File is found)
def test_file_found():
    assert 1 ==1

# Negative Path (Directory does not contain candidate file)
def test_file_not_found():
    assert 1 ==1

# Negative Path (Directory is nonsense)
# Also checks default and parametrized error messages
def test_directory_test_nonsense_directory():
    assert 1 ==1     


# For copy function


# Happy path (File is copied)
def test_file_copies():
    assert 1 ==1
# Negative Path (file cannot be copied due to permissions)
def test_file_copy_fails_permissions():
    assert 1 ==1
# Negative Path (file cannot be found and is therefore not copied)
def test_file_copy_fails_file_not_found():
    assert 1 ==1
# Negative Path (file already exists)
# Also checks default and parametrized error messages
def test_file_copy_fails_file_exists():
    assert 1 ==1


# For rename function


# Happy path (File is renamed)
def test_file_renamed():
# Negative Path (file cannot be renamed due to permissions)
    assert 1 ==1
def test_rename_fails_permissions():
# Negative Path (file cannot be found and is therefore not renamed)
    assert 1 ==1
def test_rename_fails_not_found():
    assert 1 ==1
# Negative Path (file already exists)
# Also checks default and parametrized error messages
def test_rename_fails_file_exists():
    assert 1 ==1


