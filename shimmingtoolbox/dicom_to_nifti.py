#!usr/bin/env python3
# -*- coding: utf-8

# TODO: check in unit test if dcm2bids_scaffold is installed, and also check for the required version.

from distutils.dir_util import copy_tree
import json
import numpy as np
import os
import subprocess

from shimmingtoolbox import __dir_config_dcm2bids__

# TODO: check in unit test if dcm2bids_scaffold is installed, and also check for the required version.


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

    """
    # TODO: remove temp tmp_dcm2bids (if user wants to)

    # Create the folder where the nifti files will be stored
    if not os.exists(path_nifti):
        os.makedirs(path_nifti)
    # Create bids structure for data
    # TODO; use dcm2bids as python package (no system call)
    subprocess.run(['dcm2bids_scaffold', '-o', path_nifti], check=True)
    #
    # # Copy original dicom files into nifti_path/sourcedata
    copy_tree(path_dicom, os.path.join(path_nifti, 'sourcedata'))
    #
    # # Call the dcm2bids_helper
    subprocess.run(['dcm2bids_helper', '-d', path_dicom, '-o', path_nifti], check=True)
    #
    # # Check if the helper folder has been created
    # helper_path = os.path.join(path_nifti, 'tmp_dcm2bids', 'helper')
    # if not os.path.isdir(helper_path):
    #     raise ValueError('dcm2bids_helper could not create directory helper')
    #
    # # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    # helper_file_list = os.listdir(helper_path)
    # if not helper_file_list:
    #     raise ValueError('No data to process')
    #
    # # Create list of acquisitions
    # acquisition_names = []
    # acquisition_numbers = []
    # modality = []
    #
    # # Create lists containing all acquisition names and numbers
    # for file in [file for file in helper_file_list if file.endswith(".json")]:
    #     name, ext = os.path.splitext(file)
    #     # Check for both.gz and .nii
    #     niftis = [name + ext for ext in [".nii", ".nii.gz"] if (name + ext) in helper_file_list]
    #     nifti = str(niftis[0])
    #
    #     # Read json file
    #     _, json_data = read_nii(os.path.join(helper_path, nifti))
    #
    #     # Create future folder name
    #     acquisition_numbers.append(json_data['SeriesNumber'])
    #     acquisition_names.append(json_data['SeriesDescription'])
    #     # Modality could be used as acquisition name
    #     modality.append(json_data['Modality'])
    #
    # # Remove duplicates
    # acquisition_numbers, ia = np.unique(acquisition_numbers, return_index=True)
    # acquisition_names_short = []
    # modality_short = []
    # for iAcq in ia:
    #     acquisition_names_short.append(acquisition_names[iAcq])
    #     modality_short.append(modality[iAcq])
    #
    # # Folder where the different nifti acquisitions will be stored
    # output_dir = os.path.join(path_nifti, 'code')
    #
    subprocess.run(['dcm2bids', '-d', path_dicom, '-o', path_nifti, '-p', subject_id, '-c', path_config_dcm2bids], check=True)
