from distutils.dir_util import copy_tree
import json
import numpy as np
import os
import subprocess
from .read_nii import read_nii

# TODO: check in unit test if dcm2bids_scaffold is installed, and also check for the required version.


def dicom_to_nifti(unsorted_dicom_dir, nifti_path):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        unsorted_dicom_dir (str): path to the folder where the unsorted dicom files are stored
        nifti_path (str): path to the folder where we want dcm2bids to store the nifti files

    """

    # Create the folder where the nifti files will be stored
    os.makedirs(nifti_path)

    # Create bids structure for data
    subprocess.run(['dcm2bids_scaffold', '-o', nifti_path], check=True)

    # Copy original dicom files into nifti_path/sourcedata
    copy_tree(unsorted_dicom_dir, os.path.join(nifti_path, 'sourcedata'))

    # Call the dcm2bids_helper
    subprocess.run(['dcm2bids_helper', '-d', unsorted_dicom_dir, '-o', nifti_path], check=True)

    # Check if the helper folder has been created
    helper_path = os.path.join(nifti_path, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(helper_path):
        raise ValueError('dcm2bids_helper could not create directory helper')

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(helper_path)
    if not helper_file_list:
        raise ValueError('No data to process')

    # Create list of acquisitions
    acquisition_names = []
    acquisition_numbers = []
    modality = []

    # Create lists containing all acquisition names and numbers
    for file in [file for file in helper_file_list if file.endswith(".json")]:
        name, ext = os.path.splitext(file)
        # Check for both.gz and .nii
        niftis = [name + ext for ext in [".nii", ".nii.gz"] if (name + ext) in helper_file_list]
        nifti = str(niftis[0])

        # Read json file
        _, json_data = read_nii(os.path.join(helper_path, nifti))

        # Create future folder name
        acquisition_numbers.append(json_data['SeriesNumber'])
        acquisition_names.append(json_data['SeriesDescription'])
        # Modality could be used as acquisition name
        modality.append(json_data['Modality'])

    # Remove duplicates
    acquisition_numbers, ia = np.unique(acquisition_numbers, return_index=True)
    acquisition_names_short = []
    modality_short = []
    for iAcq in ia:
        acquisition_names_short.append(acquisition_names[iAcq])
        modality_short.append(modality[iAcq])

    # Folder where the different nifti acquisitions will be stored
    output_dir = os.path.join(nifti_path, 'code')

    for iAcq in range(len(acquisition_names_short)):
        # Create a config.json file, place it in nifti_path / code
        config_file_path = create_config(output_dir, acquisition_numbers[iAcq], acquisition_names_short[iAcq], modality_short[iAcq])

        # Call dcm2bids
        participant = ''
        subprocess.run(['dcm2bids', '-d', unsorted_dicom_dir, '-o', nifti_path, '-p', participant, '-c', config_file_path], check=True)


def create_config(output_dir, acquisition_number, acquisition_name, modality):
    """ Creates a config.json file used by dcm2bids to sort the nifti files of a same acquisition into a same folder

    Args:
        output_dir (str): path to the folder where dcm2bids will store the acquisitions as a folder
        acquisition_number (int): number of the acquisition
        acquisition_name (str): name of the acquisition
        modality (str): name of the modality used during the acquisition

    Returns:
        file_path: path to the folder where the nifti and json files corresponding to the acquisition have been stored

    """
    # Define the config file name
    name = str(acquisition_number) + '_' + acquisition_name
    file_path = os.path.join(output_dir, name + '.json')

    # Create a dictionary that will be used to write a config.json file
    config = {
        "searchMethod": "fnmatch",
        "defaceTpl": "pydeface --outfile {dstFile} {srcFile}",
        "descriptions": [{
            "dataType": name,
            "SeriesDescription": name,
            "modalityLabel": modality,
            "criteria": {
                "SeriesDescription": acquisition_name,
                "SeriesNumber": str(acquisition_number)
            },
        }],
    }

    # Create a config.json file that will be used by dcm2bids to separate the nifti files
    with open(file_path, 'w') as config_file:
        json.dump(config, config_file)

    # Close file
    config_file.close()

    return file_path
