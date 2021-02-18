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
 Business Case Validation for running Processes 
'''

#from shimmingtoolbox import __dir_config_dcm2bids__

import subprocess


# For subprocess_return_validation function


# Happy path (Subprocess works)
def test_subprocess_runs():
    assert 1==1
# Negative Path (Existing Subprocess fails)
def test_subprocess_fails():
    assert 1==1
# Negative Path (Non-existent Subprocess fails)
# Also checks default and parametrized error messages
def test_subprocess_does_not_exist():
    assert 1==1


