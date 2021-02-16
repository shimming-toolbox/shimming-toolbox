#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import pytest

from shimmingtoolbox.language import English as notice
from shimmingtoolbox import __dir_testing__

# For json_load_validation function


# Happy path 1/2 (JSON file loads)
def test_file_loads:



# Happy path 2/2 (JSON file is empty)
def test_file_is_empty:



# Negative Path (JSON file does not exist)
# Also checks default and parametrized error messages
def test_file_missing:


# For json_data_valid function

# Happy path 1 (JSON has correct data)
def test_correct_data:

# Negative Path (JSON is missing fields)
# Also checks default and parametrized error messages
def test_missing_fields:




# For is_directory_nonEmpty function


# Happy path 1 (Directory contains file)
def test_directory_has_file:

# Negative Path (Directory is empty)
# Also checks default and parametrized error messages
def test_directory_empty:




# For all_with_extension function



# Happy path (Directory contains extension)
def test_has_extensions:

# Negative Path (Directory does not contain extension)
def test_no_extensions:

# Negative Path (Extension is nonsense)
def test_nonsense_extensions:

# Negative Path (Directory is nonsense)
# Also checks default and parametrized error messages
def test_nonsense_directory:


	
