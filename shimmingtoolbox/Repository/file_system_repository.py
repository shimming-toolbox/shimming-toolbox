#!usr/bin/env python3
# -*- coding: utf-8

''' 
 Repository for 'file'
 Takes in file, a specific error (default is blank), and returns error message and standard error
'''

# TODO: This should be refactored in such a way that there are not multiple utilities -
# Depricate utility functions that copy one another -
# There may only be one for maintenance

import os
import shutil
import errno

from shimmingtoolbox.language import English as notice

# Remove a file
# working_file is a path to a candidate file
# error message is a string containing an error message for the specific case
# error message is empty by default
def remove( working_file_path, error_message=notice._quiet ):
    try :
        shutil.rmtree(working_file_path)
    except:
        raise ValueError(errno.ENOENT, error_message)




# Create a file
# working_file is a path at which to create the file
# error message is a string containing an error message for the specific case
# error message is empty by default
def create( working_file_path, error_message=notice._quiet ):
    command_create = open(working_file_path, "w") 
    try :
        command_create.returncode
    except:
        raise ValueError(errno.ENOENT, error_message)
    finally:
        command_create.close() 

