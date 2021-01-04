#!usr/bin/env python3
# -*- coding: utf-8

'''
 Business Case Validation for Data 
'''

#from shimmingtoolbox import __dir_config_dcm2bids__
import goop
import language as notice
import os

'''
 Identifies if there are files in a directory
 Error, error_message, is blank by default
 Requires directory path, dir_path, as a string
'''
def is_directory_nonEmpty(directory_path, error_message = _quiet):
    file_list = os.listdir(path_helper)
    if len(file_list) < 1:
	raise ValueError(errno.ENODATA, notice.message_lang.error_message, file_list.stderr)

'''
 Identifies the list of files with a specific extension in a directory
 Default error_message is blank, may be overwritten
 Requires directory path, dir_path, as a string
 Requires file extension to search for, search_extension
'''
def all_with_extension( directory_path, extension, error_message = _quiet ):
	if sys_repository.contains_extension(directory_path, extension, error_message = _quiet ):
		file_list = []		
		search_string = String.join("*.", extension)
		for found_file in glob.glob(search_string, recursive=False):
            		file_list.append(os.path.join(path_fmap, found_file))
            		file_list = sorted(file_list)
		return file_list
	raise ValueError(errno.ENODATA, notice.message_lang.error_message)

'''
 Identifies if a data file (json) can be loaded
 Default error_message is blank, may be overwritten
 Requires file path
 returns the json object wherein there are no errors
'''
#TODO: could probably grab the stack trace as well
def json_load_validation( file_to_load, error_message = _quiet ):
	try:
		json = json.load(file_to_load)
		return json
	except:
		raise ValueError( errno.ENODATA, notice.message_lang.error_message )

'''
 Identifies if a json file has the correct data fields for phase data
 Default error_message is blank, may be overwritten
 json_data must be a json object
 Requires file path
'''
#TODO: could probably grab the stack trace as well
def json_data_valid( json_data, error_message = _quiet ):

	# Validate JSON object
	json_data = json_load_validation( file_to_load )
        
        # Make sure it is a phase data and that the keys EchoTime1 and EchoTime2 are defined and that
        # sequenceName's last digit is 2 (refers to number of echoes when using dcm2bids)
        try: True == (('ImageType' in json_data) and \
		('P' in json_data['ImageType']) and \
		('EchoTime1' in json_data) and \
		('EchoTime2' in json_data) and \
                ('SequenceName' in json_data) and \
		(int(json_data['SequenceName'][-1]) == 2))

	except:
		raise ValueError( errno.ENODATA, notice.message_lang._json_missing_fields )
	


	
