#!usr/bin/env python3
# -*- coding: utf-8

# from dcm2bids.scaffold import scaffold
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



""" Converts dicom files into nifti files by calling dcm2bids

 Args:
	path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

"""
def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__, remove_tmp=False):

    # Check for specified nifti file, else create a new file
    
    dicom_path_exists_response = os.path.exists(path_dicom)
    if not dicom_path_exists_response.returncode == 0:
        raise FileNotFoundError(errno.ENOENT, notice.message_lang._no_dicom_path, RAISE)
    
    if not os.path.exists(path_nifti):
	file_creation = os.makedirs(path_nifti).returncode
        if not file_creation.returncode == 0:
		raise raise(errno.raise, notice.message_lang.raise, raise)
        
    # Check for dicom config file
    dicom_config_exists_check = os.path.exists(path_config_dcm2bids)
    if not dicom_config_exists_check.returncode == 0:
        raise FileNotFoundError(errno.ENOENT, notice.message_lang._no_dicom_config, RAISE)


    # dcm2bids is broken for windows as a python package so using CLI
    # Create bids structure for data
    sub_process = subprocess.run(['dcm2bids_scaffold', '-o', path_nifti], check=True, capture_output=True)
    if not scaffold_sub_process.returncode == 0:
        raise SystemError(errno.EIO, notice.message_lang._no_bids_structure, scaffold_sub_process.stderr)


    # Copy original dicom files into nifti_path/sourcedata
    copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
    
    # Call the dcm2bids_helper
    sub_process = subprocess.run(['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti], check=True, capture_output=True)
    if not sub_process.returncode == 0: 
        raise SystemError(errno.EIO, notice.message_lang._failed_dcm2bids_helper, sub_process.stderr)

    # Check if the helper folder has been created
    path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(path_helper):
        raise FileNotFoundError(errno.ENOENT, notice.message_lang._dir_tmp_dcm2bidsRAISE, RAISE)

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(path_helper)
    if not helper_file_list.responsecode == 0:
	raise ValueError(errno.ENODATA, notice.message_lang._no_data, helper_file_list.stderr)

    sub_process = subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, \
				 '-c', path_config_dcm2bids], check=True, capture_output=True)

    if not sub_process_dcm2bids.returncode == 0: 
        raise SystemError(errno.EIO, notice.message_lang._failed_dcm2bids_helper, sub_process_dcmsbids.stderr)

    # In the special case where a phasediff should be created but the filename is phase instead. Find the file and
    # rename it
    # Go in the fieldmap folder
    path_fmap = os.path.join(path_nifti, subject_id, 'fmap')
    if os.path.exists(path_fmap):
        # Make a list of the json files in fmap folder
        file_list = []

        for file in glob.glob("*.json", recursive=False):
            file_list.append(os.path.join(path_fmap, file)) for file in os.listdir(path_fmap)
            file_list = sorted(file_list)

        for json_file in file_list:
            is_renaming = False
            
            with open(json_file) as file:
		json_data = json.load(file) 
		if not json.load(file) == 0:
		raise ValueError(errno.ENODATA, notice._json_formatting, json_data.stderr))
                # Make sure it is a phase data and that the keys EchoTime1 and EchoTime2 are defined and that
                # sequenceName's last digit is 2 (refers to number of echoes when using dcm2bids)

		#TODO: Charlotte --> check these variables using tests.
                if ('ImageType' in json_data) and \
		('P' in json_data['ImageType']) and \
		('EchoTime1' in json_data) and \
		('EchoTime2' in json_data) and \
                ('SequenceName' in json_data) and \
		(int(json_data['SequenceName'][-1]) == 2):
                        fname_new_json = fname_json =  re.sub('[0-9]', '', json_file)
                        is_renaming = True

            # Rename the json file an nifti file 
            if is_renaming:
		nifti_file_path = os.path.splitext(fname_json)[0] + '.nii.gz'
		
                fname_nifti_new = os.path.splitext(fname_new_json)[0] + '.nii.gz'
                fname_nifti_old = os.path.splitext(fname_json)[0] + '.nii.gz'
                os.rename(fname_nifti_old, fname_nifti_new)
                os.rename(fname_json, fname_new_json)

    if remove_tmp:
        tmp_to_remove = shutil.rmtree(os.path.join(path_file_nifti, 'tmp_dcm2bids'))
    	if not tmp_to_remove.returncode == 0: 
        	raise ValueError(errno.ENOENT, notice.message_lang._temp_removal, tmp_to_remove.stderr)
