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


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__, remove_tmp=False):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

    """


    #TODO Charlotte make this in config!!
    #TODO Charlotte check error types, you're wrong all over
    # Check for specified nifti file, else create a new file
    if not os.path.exists(path_dicom):
        raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._no_dicom_path, path_config_dicom)
    if not os.path.exists(path_nifti):
        os.makedirs(path_nifti)

    # Check for dicom config file
    if not os.path.exists(path_config_dcm2bids):
        raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._no_dicom_config, path_config_dcm2bids)


    # dcm2bids is broken for windows as a python package so using CLI
    # Create bids structure for data
    sub_process = subprocess.run(['dcm2bids_scaffold', '-o', path_nifti], check=True)
    if not sub_process.<ERROR CODE> == 0: #TODO CHARLOTTE there's a python thingy here i need to check itches wrong
        raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._no_bids_structure, sub_process.<ERROR STATEMENT>)


    # Copy original dicom files into nifti_path/sourcedata
    copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
    
    # Call the dcm2bids_helper
    sub_process = subprocess.run(['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti], check=True)
    if not sub_process.<ERROR CODE> == 0: #TODO CHARLOTTE there's a python thingy here i need to check, itches wrong
        raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._failed_dcm2bids_helper, sub_process.<ERROR STATEMENT>)

    # Check if the helper folder has been created
    path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(path_helper):
        raise ValueError(_dcm2bids_helper_creation)

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(path_helper)
    if not helper_file_list:
        raise ValueError(_no_data)

    sub_process = subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, '-c', path_config_dcm2bids],
                   check=True)
    if not sub_process.<ERROR CODE> == 0: 
        raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._failed_dcm2bids_helper, sub_process._no_dcm2bids)

    # In the special case where a phasediff should be created but the filename is phase instead. Find the file and
    # rename it
    # Go in the fieldmap folder
    path_fmap = os.path.join(path_nifti, subject_id, 'fmap')
    if os.path.exists(path_fmap): #TODO CHARLOTTE ADD AN ELSE FOR THIS BEAUTY
        # Make a list of the json files in fmap folder
        file_list = []

        for file in glob.glob("*.json", recursive=False):
            file_list.append(os.path.join(path_fmap, file)) for file in os.listdir(path_fmap)
            file_list = sorted(file_list)

        for fname_json in file_list:
            is_renaming = False
            # Open the json file TO DO CHARLOTTE ADD AN ELSE
            with open(fname_json) as json_file:  
                json_data = json.load(json_file)
                # Make sure it is a phase data and that the keys EchoTime1 and EchoTime2 are defined and that
                # sequenceName's last digit is 2 (refers to number of echoes when using dcm2bids)


#TODO Charlotte clean this up.
                if ('ImageType' in json_data) and ('P' in json_data['ImageType']) and \
                   ('EchoTime1' in json_data) and ('EchoTime2' in json_data) and \
                   ('SequenceName' in json_data) and (int(json_data['SequenceName'][-1]) == 2):
                        fname_new_json = fname_json =  re.sub('[0-9]', '', fname)
                        is_renaming = True
            # Rename the json file an nifti file 
            if is_renaming:
                #TODO: Charlotte I dont't like splittext
                if os.path.exists(os.path.splitext(fname_json)[0] + '.nii.gz'):
                    fname_nifti_new = os.path.splitext(fname_new_json)[0] + '.nii.gz'
                    fname_nifti_old = os.path.splitext(fname_json)[0] + '.nii.gz'
                    os.rename(fname_nifti_old, fname_nifti_new)
                    os.rename(fname_json, fname_new_json)

    if remove_tmp:
        removal_tmp = shutil.rmtree(os.path.join(path_nifti, 'tmp_dcm2bids'))#TODO CHARLOTTE gentalize the raise
    	if not removal_tmp.<ERROR CODE> == 0: 
        	raise FileNotFoundError(errno.ENOENT, notice.<CONFIG VARIABLE>._temp_removal, sub_process._no_dcm2bids)
