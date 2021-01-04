#!usr/bin/env python3
# -*- coding: utf-8

'''
 Business Case Validation for running Processes 
'''

#from shimmingtoolbox import __dir_config_dcm2bids__
import language as notice
import subprocess


''' 
 Identifies if a subprocess provides the expected return value for a particular business case
 Process_options is an array of subprocess options. See: https://docs.python.org/3/library/subprocess.html
 expected_value is the numerical return value expected by the subprocess
 error_message may be left blank for the default ""
'''
subprocess_return_validation( expected_value = 0, subprocess_options, error_message = _quiet ):
    process_response = subprocess.run( subprocess_options ), check=True, capture_output=True)
    if not process_response.returncode == expected_value:
        raise SystemError(errno.EIO, notice.message_lang.error_message, process_response.stderr)

