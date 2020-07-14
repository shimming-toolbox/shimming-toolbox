import os
import numpy as np
from .read_nii import read_nii


def dicom_to_nifti(unsorted_dicom_dir, nifti_path):
    """
    TODO: indicate the purpose of this function
    :param unsorted_dicom_dir:
    :param nifti_path:
    :return:
    """

    os.mkdir(nifti_path)
    print(unsorted_dicom_dir)
    print(nifti_path)

    # Check for which "which" to use depending on OS
    if os.name == 'nt':  # If OS is Windows
        which = 'where'
        copy = 'copy'
    else:
        which = 'which'
        copy = 'cp -r'

    # Make sur dcm2niix is installed
    if os.system(which + ' dcm2niix') != 0:
        print('Error: dcm2niix is not installed.')

    # Make sure dcm2bids is installed
    if os.system(which + ' dcm2bids') != 0:
        raise Exception('Error: dcm2bids is not installed.')

    # Create bids structure for data
    participant = ''
    if os.system('dcm2bids_scaffold -o ' + nifti_path) != 0:
        raise Exception('Error: dcm2bids_scaffold')

    # Add original data to nifti_path/sourcedata
    if os.system(copy + ' ' + unsorted_dicom_dir + ' ' + os.path.join(nifti_path, 'sourcedata')) != 0:
        raise Exception('Error: copy')

    # Call the dcm2bids_helper
    if os.system('dcm2bids_helper -d ' + unsorted_dicom_dir + ' -o ' + nifti_path) != 0:
        raise Exception('Error: dcm2bids_helper')

    # Check if there is data
    helper_path = os.path.join(nifti_path, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(helper_path):
        raise Exception('Error: dcm2bids_helper could not create directory helper')

    # Make sure there is data in nifti_path / tmp_dcm2bids / helper
    helper_file_list = os.listdir(helper_path)
    if not helper_file_list:
        raise Exception('No data to process')

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
                print('Nifti file ' + name + ' not found')

            nifti_file = helper_file_list[nifti_index]
            # Read json file
            _, _, jsonInfo = read_nii(os.path.join(helper_path, nifti_file))

            # Create future folder name
            acquisition_numbers.append(jsonInfo['SeriesNumber'])
            acquisition_names.append(jsonInfo['SeriesDescription'])
            # Modality could be used as acquisition name
            modality.append(jsonInfo['Modality'])

    # Remove
    acquisition_numbers, ia = np.unique(acquisition_numbers, return_index=True)
    acquisition_names_short = []
    modality_short = []
    for iAcq in ia:
        acquisition_names_short.append(acquisition_names[iAcq])
        modality_short.append(modality[iAcq])

    # Folder where the niftis will be stored
    output_dir = os.path.join(nifti_path, 'code')

    for iAcq in range(len(acquisition_names_short)):
        # Create a config.json file, place it in nifti_path / code
        config_file_path = create_config(output_dir, acquisition_numbers[iAcq], acquisition_names_short[iAcq], modality_short[iAcq])

        # Call dcm2bids
        if os.system('dcm2bids -d "' + unsorted_dicom_dir + '"' ' -o '  '"' + nifti_path + '"' ' -p '  '"' + participant + '"' ' -c ' + config_file_path) != 0:
            print('Error: dcm2bids failed')


def create_config(output_dir, acquisition_number, acquisition_name, modality):

    # Create file name
    name = str(acquisition_number) + '_' + str(acquisition_name) + '.json'
    file_path = os.path.join(output_dir, name)

    # Create file
    fid = open(file_path, "w+")

    # Write to file
    fid.write('{\n')
    fid.write('    "searchMethod": "fnmatch",\n')
    fid.write('    "defaceTpl": "pydeface --outfile {dstFile} {srcFile}",\n')
    fid.write('    "descriptions": [\n')
    fid.write('        {\n')
    fid.write('            "dataType": "' + name + '",\n')
    fid.write('            "SeriesDescription": "' + name + '",\n')
    fid.write('            "modalityLabel": "' + modality + '",\n')
    fid.write('            "criteria": {\n')
    fid.write('                "SeriesDescription": "' + acquisition_name + '",\n')
    fid.write('                "SeriesNumber": "' + str(acquisition_number) + '"\n')
    fid.write('              }\n')
    fid.write('        }\n')
    fid.write('    ]\n')
    fid.write('}\n')

    # Close file
    fid.close()

    return file_path
