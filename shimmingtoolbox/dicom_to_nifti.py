from distutils.dir_util import copy_tree
import numpy as np
import os
import subprocess
from .read_nii import read_nii


def dicom_to_nifti(unsorted_dicom_dir, nifti_path):
    """ Converts dicom files into nifti files following bids convention

    Args:
        unsorted_dicom_dir: path to the folder where the unsorted dicom files are stored
        nifti_path: path to the folder where we want dcm2bids to store the nifti files

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

    # Create list containing all files
    for iFile in range(len(helper_file_list)):
        # If it's a json file
        name, ext = os.path.splitext(helper_file_list[iFile])
        if ext == '.json':
            # Check for both .nii.gz and .nii
            if name + '.nii.gz' in helper_file_list:
                nifti_index = helper_file_list.index(name + '.nii.gz')
            elif name + '.nii' in helper_file_list:
                nifti_index = helper_file_list.index(name + '.nii')
            else:
                raise TypeError('Nifti file "' + name + '" not found')

            nifti_file = helper_file_list[nifti_index]
            # Read json file
            _, _, json_data = read_nii(os.path.join(helper_path, nifti_file))

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

    # Define the config file name
    name = str(acquisition_number) + '_' + str(acquisition_name) + '.json'
    file_path = os.path.join(output_dir, name)

    # Create file
    config_file = open(file_path, "w+")

    # Write to file
    config_file.write('{\n')
    config_file.write('    "searchMethod": "fnmatch",\n')
    config_file.write('    "defaceTpl": "pydeface --outfile {dstFile} {srcFile}",\n')
    config_file.write('    "descriptions": [\n')
    config_file.write('        {\n')
    config_file.write('            "dataType": "' + name + '",\n')
    config_file.write('            "SeriesDescription": "' + name + '",\n')
    config_file.write('            "modalityLabel": "' + modality + '",\n')
    config_file.write('            "criteria": {\n')
    config_file.write('                "SeriesDescription": "' + acquisition_name + '",\n')
    config_file.write('                "SeriesNumber": "' + str(acquisition_number) + '"\n')
    config_file.write('              }\n')
    config_file.write('        }\n')
    config_file.write('    ]\n')
    config_file.write('}\n')

    # Close file
    config_file.close()

    return file_path
