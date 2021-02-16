#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest

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
def test_directory_has_extension:

# Negative Path (Directory does not contain extension)
def test_directory_no_extension:

# Negative Path (Extension is nonsense)
def test_nonsense_extensions:

# Negative Path (Directory is nonsense)
# Also checks default and parametrized error messages
def test_nonsense_directory:




# For isFound function


# Happy path (File is found)
def test_file_found


# Negative Path (Directory does not contain candidate file)
def test_file_not_found


# Negative Path (Directory is nonsense)
# Also checks default and parametrized error messages
def test_directory_test_nonsense_directory:
        


# For copy function


# Happy path (File is copied)
def test_file_copies

# Negative Path (file cannot be copied due to permissions)
def test_file_copy_fails_permissions:

# Negative Path (file cannot be found and is therefore not copied)
def test_file_copy_fails_file_not_found:

# Negative Path (file already exists)
# Also checks default and parametrized error messages
def test_file_copy_fails_file_exists:



# For rename function


# Happy path (File is renamed)
def test_file_renamed:
# Negative Path (file cannot be renamed due to permissions)

def test_rename_fails_permissions:
# Negative Path (file cannot be found and is therefore not renamed)

def test_rename_fails_not_found:

# Negative Path (file already exists)
# Also checks default and parametrized error messages
def test_rename_fails_file_exists:



