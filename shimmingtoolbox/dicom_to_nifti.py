#!usr/bin/env python3
# -*- coding: utf-8

from distutils.dir_util import copy_tree
import json
import os
import subprocess
# from dcm2bids.scaffold import scaffold
import shutil

from shimmingtoolbox import __dir_config_dcm2bids__
from shimmingtoolbox.utils import create_output_dir


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
    create_output_dir(path_nifti)

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

    # In the special case where a phasediff should be created but the filename is phase instead. Find the file and
    # rename it
    # Go in the fieldmap folder
    path_fmap = os.path.join(path_nifti, subject_id, 'fmap')
    if os.path.exists(path_fmap):
        # Make a list of the json files in fmap folder
        file_list = []
        [file_list.append(os.path.join(path_fmap, f)) for f in os.listdir(path_fmap)
         if os.path.splitext(f)[1] == '.json']
        file_list = sorted(file_list)

        for fname_json in file_list:
            is_renaming = False
            # Open the json file
            with open(fname_json) as json_file:
                json_data = json.load(json_file)
                # Make sure it is a phase data and that the keys EchoTime1 and EchoTime2 are defined and that
                # sequenceName's last digit is 2 (refers to number of echoes when using dcm2bids)
                if ('ImageType' in json_data) and ('P' in json_data['ImageType']) and \
                   ('EchoTime1' in json_data) and ('EchoTime2' in json_data) and \
                   ('SequenceName' in json_data) and (int(json_data['SequenceName'][-1]) == 2):
                    # Make sure it is not already named phasediff
                    if len(os.path.basename(fname_json).split(subject_id, 1)[-1].rsplit('phasediff', 1)) == 1:
                        # Split the filename in 2 and remove phase
                        file_parts = fname_json.rsplit('phase', 1)
                        if len(file_parts) == 2:
                            # Stitch the filename back together making sure to remove any digits that could be after
                            # 'phase'
                            digits = '0123456789'
                            fname_new_json = file_parts[0] + 'phasediff' + file_parts[1].lstrip(digits)
                            is_renaming = True
            # Rename the json file an nifti file
            if is_renaming:
                if os.path.exists(os.path.splitext(fname_json)[0] + '.nii.gz'):
                    fname_nifti_new = os.path.splitext(fname_new_json)[0] + '.nii.gz'
                    fname_nifti_old = os.path.splitext(fname_json)[0] + '.nii.gz'
                    os.rename(fname_nifti_old, fname_nifti_new)
                    os.rename(fname_json, fname_new_json)

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
    #     subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, '-c', path_config_dcm2bids],
    #     check=True)
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
        shutil.rmtree(os.path.join(path_nifti, 'tmp_dcm2bids'))
