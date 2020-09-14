#!usr/bin/env python3
# -*- coding: utf-8

from distutils.dir_util import copy_tree
import json
import numpy as np
import os
import sys
import subprocess
import dcm2bids
# from dcm2bids.scaffold import scaffold

from shimmingtoolbox import __dir_config_dcm2bids__

def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__, remove_tmp=False):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

    """

    # Create the folder where the nifti files will be stored
    if not os.path.exists(path_dicom):
        raise FileNotFoundError("No dicom path found")
    if not os.path.exists(path_config_dcm2bids):
        raise FileNotFoundError("No dcm2bids config file found")
    if not os.path.exists(path_nifti):
        os.makedirs(path_nifti)

    # dcm2bids is broken for windows as a python package so using CLI
    # Create bids structure for data
    subprocess.run(['dcm2bids_scaffold', '-o', path_nifti], check=True)

    #
    # # Copy original dicom files into nifti_path/sourcedata
    copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
    #
    # # Call the dcm2bids_helper
    subprocess.run(['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti], check=True)
    #
    # Check if the helper folder has been created
    path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(path_helper):
        raise ValueError('dcm2bids_helper could not create directory helper')

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(path_helper)
    if not helper_file_list:
        raise ValueError('No data to process')

    subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, '-c', path_config_dcm2bids],
                   check=True)

    # if 'win' in sys.platform:
    #     # dcm2bids is broken for windows as a python package so using CLI
    #     # Create bids structure for data
    #     subprocess.run(['dcm2bids_scaffold', '-o', path_nifti], check=True)
    #
    #     #
    #     # # Copy original dicom files into nifti_path/sourcedata
    #     copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
    #     #
    #     # # Call the dcm2bids_helper
    #     subprocess.run(['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti], check=True)
    #     #
    #     # Check if the helper folder has been created
    #     path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    #     if not os.path.isdir(path_helper):
    #         raise ValueError('dcm2bids_helper could not create directory helper')
    #
    #     # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    #     helper_file_list = os.listdir(path_helper)
    #     if not helper_file_list:
    #         raise ValueError('No data to process')
    #
    #     subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, '-c', path_config_dcm2bids], check=True)
    # else:
    #     bids_info = dcm2bids.Dcm2bids([path_dicom], subject_id, path_config_dcm2bids, path_nifti)
    #     scaffold()
    #
    #     path_helper = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    #     if not os.path.isdir(path_helper):
    #         raise ValueError('dcm2bids_helper could not create directory helper')
    #     helper_file_list = os.listdir(path_helper)
    #     if not helper_file_list:
    #         raise ValueError('No data to process')
    #
    #     bids_info.run()

    if remove_tmp:
        os.removedirs(os.path.join(path_dicom, 'tmp_dcm2bids'))
