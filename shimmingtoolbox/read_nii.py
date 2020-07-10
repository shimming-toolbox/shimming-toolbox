import json
import math
import nibabel as nib
import os


def read_nii(niiPath):
    """

    Args:
        niiPath (str): Path to the nifti file

    Returns:
        image:
        info:
        jsonData:
    """

    info = nib.load(niiPath)
    image = info.get_fdata()

    # `extractBefore` should get the correct filename in both.nii and.nii.gz cases
    jsonPath = niiPath.split('.nii')[0] + '.json'

    if os.path.isfile(jsonPath):
        jsonData = json.load(open(jsonPath))
    else:
        jsonData = []
        print('Warning: No json file has been found')

    # NOTE: nib.load automatically scales the nifti and replaces scl_inter and scl_slope with 'nan'
    # See the "Data scaling" section in https://nipy.org/nibabel/nifti_images.html
    if ('Manufacturer' in jsonData) and (jsonData['Manufacturer'] == 'Siemens') \
            and (check_json_image_type(jsonData) == 'phase'):

        image, info = convert_siemens_phase(image, info)

    return image, info, jsonData

# -----------------------------------------------------------------------------
# Local functions
# -----------------------------------------------------------------------------


def check_json_image_type(jsonData):
    imgType = []

    if not jsonData:
        print('Warning: The image type has not been found because the json file is empty')
        return imgType

    assert isinstance(jsonData, dict)  # Check that jsonData is a dictionary

    if 'ImageType' in jsonData:
        isPhase = "P" in jsonData['ImageType']
        isMag = "M" in jsonData['ImageType']

        if isPhase and isMag:
            # Both true: json file and / or DICOM issue(hopefully this doesn't occur?)
            print('Warning: Ambiguous ImageType entry in json file: Indicates magnitude AND phase?')
        elif isPhase:
            imgType = 'phase'
        elif isMag:
            imgType = 'magnitude'
        else:
            imgType = 'unknown'
            if ('Manufacturer' in jsonData) and (jsonData['Manufacturer'] != 'Siemens'):
                print('Warning: Unknown image type. Possibly due to images sourced from non-Siemens MRI')

        return imgType


def convert_siemens_phase(image, info):

    PHASE_SCALING_SIEMENS: int = 4096

    infoConverted = info
    imgConverted = image * (math.pi / PHASE_SCALING_SIEMENS)

    # Update header: 16 is the NIfTI code for float; bitpix = number of bits
    infoConverted.header['datatype'] = 16
    infoConverted.header['bitpix'] = 32

    return imgConverted, infoConverted
