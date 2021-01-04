#!usr/bin/env python3
# -*- coding: utf-8

''' 
 Repository for 'file'
 Takes in file, a specific error (default is blank), and returns error message and standard error
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


def found( working_file, error_message=_quiet ):
	command_find = os.path.exists(working_file)
	if not command_find.returncode == 0:
        	raise FileNotFoundError(errno.ENOENT, notice.message_lang.error_message, find_check.stderr)

def copied( working_file, new_file, error_message=_quiet ):
	command_copy = copy_tree(working_file, new_file)
    	if not command_copy.returncode == 0: 
        	raise ValueError(errno.ENOENT, notice.message_lang.error_message, command_copy.stderr)

def rename( working_file, new_name, error_message=_quiet ):
	command_rename = os.rename(working_file, new_name)
    	if not command_rename.returncode == 0: 
        	raise ValueError(errno.ENOENT, notice.message_lang.error_message, command_rename.stderr)

def remove( working_file, error_message=_quiet ):
        command_remove = shutil.rmtree(os.path.join(path_file_nifti, 'tmp_dcm2bids'))
    	if not command_remove.returncode == 0: 
        	raise ValueError(errno.ENOENT, notice.message_lang.error_message, command_remove.stderr)
