#!usr/bin/env python3
# -*- coding: utf-8

'''
 Business Case Validation for the file system
'''
# TODO: This should be refactored in such a way that there are not multiple utilities -
# Depricate utility functions that copy one another -
# There may only be one for maintenance
import goop
import errno
import os

from shimmingtoolbox.language import English as notice


'''
 Identifies if there is at least one file with the provided extension
 Returns true or false
 Requires directory path, dir_path, as a string
 Requires extension as a string
'''
def does_contain_extension( directory_path, extension ): 
    count = glob.glob("*.json", recursive=False)
    
    if count < 1 :
        return True
    else :
        return False
        
# Find if a file exists
# working_file is a path to a candidate file
# error message is a string containing an error message for the specific case
# error message is empty by default
def isFound( working_file, error_message=notice._quiet ):
	command_find = os.path.exists(working_file)
	if not command_find == True:
		raise NameError(errno.ENOENT, notice.error_message)
        	
# Copy a file
# working_file is a path to a candidate file
# error message is a string containing an error message for the specific case
# error message is empty by default
def copy( working_file, new_file, error_message=notice._quiet ):
    if not copy2(working_file, new_file):
        raise ValueError(errno.ENOENT, notice.error_message, command_copy.stderr)
        
        
import os, shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
        
# Rename a File
# working_file is a path to the file that requires a name change
# new_name is the new name for working_file
# error message is a string containing an error message for the specific case
# error message is empty by default
def rename( working_file, new_name, error_message=notice._quiet ):
    command_rename = os.rename(working_file, new_name)
    if not command_rename.returncode == 0: 
        raise ValueError(errno.ENOENT, notice.error_message, command_rename.stderr)


