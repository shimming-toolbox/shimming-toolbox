import os
import numpy as np
from .read_nii import read_nii


def dicom_to_nifti(unsortedDicomDir, niftiPath):

    #os.mkdir(niftiPath)
    print(unsortedDicomDir)
    print(niftiPath)

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
    if os.system('dcm2bids_scaffold -o ' + niftiPath) != 0:
        raise Exception('Error: dcm2bids_scaffold')

    # Add original data to niftiPath/sourcedata
    if os.system(copy + ' ' + unsortedDicomDir + ' ' + os.path.join(niftiPath, 'sourcedata')) != 0:
        raise Exception('Error: copy')

    # Call the dcm2bids_helper
    if os.system('dcm2bids_helper -d ' + unsortedDicomDir + ' -o ' + niftiPath) != 0:
        raise Exception('Error: dcm2bids_helper')

    # Check if there is data
    helperPath = os.path.join(niftiPath, 'tmp_dcm2bids', 'helper')
    if not os.path.isdir(helperPath):
        raise Exception('Error: dcm2bids_helper could not create directory helper')

    # Make sure there is data in niftiPath / tmp_dcm2bids / helper
    helperfileList = os.listdir(helperPath)
    if not helperfileList:
        raise Exception('No data to process')

    # Create list of acquisitions
    acquisitionNames = list()
    acquisitionNumbers = list()
    iAcq = 0

    # Create list containing all files
    for iFile in range(len(helperfileList)):
        # If it's a json file
        name, ext = os.path.splitext(helperfileList[iFile])
        if ext == '.json':
            # Check for both.gz and .nii
            booleanList = (np.in1d(helperfileList, name + '.nii') + np.in1d(helperfileList, name + '.nii.gz')).tolist()
            niftiFile = helperfileList[booleanList.index(1)]  # Returns an exception if no True is found

        # Read json file
        _, _, jsonInfo = read_nii(os.path.join(helperPath, niftiFile))
#     iAcq = iAcq + 1;
#
#     # Create future folder name acquisitionNumbers
#     {iAcq, 1} = sprintf('#03d', jsonInfo.SeriesNumber);
#     acquisitionNames
#     {iAcq, 1} = jsonInfo.SeriesDescription;
#     # ** ** ** *Modality could be acquisition name ** ** ** **
#     modality{iAcq, 1} = jsonInfo.Modality;
#
# # Remove duplicates(stable is specified to keep the same order)
# [acquisitionNumbers, ia] = unique(acquisitionNumbers, 'stable');
# acquisitionNames = acquisitionNames(ia);
# modality = modality(ia);
#
# # For every acqs,
# outputDir = fullfile(niftiPath, 'code');
#
# clear iAcq
# for iAcq = 1:length(acquisitionNames):
#     # create config file, place in niftiPath / code
#     configFilePath = createConfig(outputDir, acquisitionNumbers{iAcq}, acquisitionNames{iAcq}, modality{iAcq});
#
#     # call dcm2bids
#     if system(['dcm2bids -d "' unsortedDicomDir '"' ' -o '  '"' niftiPath '"' ' -p '  '"' participant '"' ' -c 'configFilePath]) ~= 0:
#         error('dcm2bids failed')
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Local functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# def filePath = createConfig(outputDir, acquisitionNumber, acquisitionName, modality)
#
# # Create names
# name = [acquisitionNumber '_' acquisitionName];
# ext = '.json';
# filePath = fullfile(outputDir, [name ext]);
#
# # Create file
# fid = fopen(filePath, 'w');
# if fid == -1:
#     error('could not create config file')
#
# # Write to file
# fprintf(fid, '{\n');
# fprintf(fid, '    "searchMethod": "fnmatch",\n');
# fprintf(fid, '    "defaceTpl": "pydeface --outfile {dstFile} {srcFile}",\n');
# fprintf(fid, '    "descriptions": [\n');
# fprintf(fid, '        {\n');
# fprintf(fid, '            "dataType": "#s",\n', name);
# fprintf(fid, '            "SeriesDescription": "#s",\n', name);
# fprintf(fid, '            "modalityLabel": "#s",\n', modality);
# fprintf(fid, '            "criteria": {\n');
# fprintf(fid, '                "SeriesDescription": "#s",\n', acquisitionName);
# fprintf(fid, '                "SeriesNumber": "#s"\n', num2str(str2num(acquisitionNumber)));
# fprintf(fid, '              }\n');
# fprintf(fid, '        }\n');
# fprintf(fid, '    ]\n');
# fprintf(fid, '}\n');
#
# # TODO: Add criteria in each file to add phase and magnitude seperation?
#
# # Close file
# fclose(fid);
