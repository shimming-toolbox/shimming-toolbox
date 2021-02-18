#!usr/bin/env python3
# -*- coding: utf-8

# from dcm2bids.scaffold import scaffold
from distutils.dir_util import copy_tree

from shimmingtoolbox import __dir_config_dcm2bids__

from shimmingtoolbox.Domain import data_domain as d_data
from shimmingtoolbox.Domain import file_system_domain as d_file_system
from shimmingtoolbox.Domain import process_domain as d_process
from shimmingtoolbox.Repository import file_system_repository as r_file_system
from shimmingtoolbox.language import English as notice


import goop
import json
import numpy as np
import os
import re
import sys
import subprocess
import dcm2bids
import shutil
import errno


""" Converts dicom files into nifti files by calling dcm2bids

 Args:
	path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

"""
def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__, remove_tmp=False):

    # Check for specified nifti file, else create a new file
    d_file_system.isFound( path_dicom, error_message=notice._no_dicom_path )
    r_file_system.create( path_nifti, notice._no_nifty_file )
   
    # Check for dicom config file
    d_file_system.isFound( path_config_dcm2bids, notice._no_dicom_config )

    # dcm2bids is broken for windows as a python package so using CLI
    # Create bids structure for data
    subprocess_arguments = ['dcm2bids_scaffold', '-o', 'path_nifti']
    error_message = notice._no_bids_structure

    expected_value = 0
    d_process.subprocess_return_validation( subprocess_arguments, expected_value, error_message )    

    # Copy original dicom files into nifti_path/sourcedata
    # TODO: This copy_tree should be put intp file system domain 
    # And with it use: notice._copy_dicom_failure
    copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
   
    # Call the dcm2bids_helper
    subprocess_arguments = None
    error_message = None

    subprocess_arguments = ['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti]
    error_message = notice._failed_dcm2bids_helper

    expected_value = 0
    d_process.subprocess_return_validation( subprocess_arguments, expected_value, error_message )    


    # Check if the helper folder has been created
    path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    d_file_system.isFound( path_helper, error_message=notice._dcm2bids_helper_creation)

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(path_helper)
    if not helper_file_list.responsecode == 0:
        raise ValueError(errno.ENODATA, notice._no_data)

    sub_process = subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, \
				 '-c', path_config_dcm2bids], check=True, capture_output=True)

    if not sub_process_dcm2bids.returncode == 0: 
        raise SystemError(errno.EIO, notice._failed_dcm2bids_helper, sub_process_dcmsbids.stderr)

    # In the special case where a phasediff should be created but the filename is phase instead. Find the file and
    # rename it
    # Go in the fieldmap folder
    path_fmap = os.path.join(path_nifti, subject_id, 'fmap')
    if os.path.exists(path_fmap):
        # Make a list of the json files in fmap folder
        file_list = []

        for file in glob.glob("*.json", recursive=False):
            file_list.append(os.path.join(path_fmap, file))
            file_list = sorted(file_list)
            
            with open(json_file) as file:
                if not json.load(file):
                    raise SystemError(errno.EIO, notice._json_formatting)

        json_data =  d_data.json_load_validation( file_to_load, error_message = notice._quiet )

        d_data.json_data_valid( json_data, error_message = notice._quiet )

        fname_new_json = fname_json =  re.sub('[0-9]', '', file_to_load)

        nifti_file_path = os.path.splitext(fname_json)[0] + '.nii.gz'
        fname_nifti_new = os.path.splitext(fname_new_json)[0] + '.nii.gz'
        d_file_system.rename( nifti_file_path, fname_nifti_new )

    if remove_tmp:
        path_to_file = os.path.join(path_nifti, 'tmp_dcm2bids')
        r_file_system.remove( path_to_file, notice._temp_removal )
