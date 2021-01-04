#!usr/bin/env python3
# -*- coding: utf-8

'''
 Business Case Validation for Data 
'''

from distutils.dir_util import copy_tree
from shimmingtoolbox import __dir_config_dcm2bids__

import goop
import json
import language as notice
import numpy as np
import os
import re
import sys
import subprocess
import dcm2bids
import shutil

'''
 Identifies if there are files in a directory for a particular business case
 Error, error_message, is blank by default
 Requires directory path, dir_path, as a string
'''
def is_directory_nonEmpty(directory_path, error_message = _quiet):
    file_list = os.listdir(path_helper)
    if len(file_list) < 0:
	raise ValueError(errno.ENODATA, notice.message_lang.error_message, file_list.stderr)

'''
 Identifies if there are files with a specific extension in a directory for a particular business case
 Default, error_message is blank, may be overwritten
 Requires directory path, dir_path, as a string
 Requires file extension to search for, search_extension
'''
def contains_extension(): 

