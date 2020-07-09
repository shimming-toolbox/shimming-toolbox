import json
import math
import nibabel as nib
import numpy as np
import os


def read_nii(niiFile, rescaling='auto'):
    assert (rescaling in "off" "basic" "auto"), 'Invalid assignment to rescale: Value must be "off", "basic", or "auto"'

    info = nib.load(niiFile)
    image = info.get_fdata()

    # `extractBefore` should get the correct filename in both.nii and.nii.gz cases
    jsonPath = niiFile.split('.nii')[0] + '.json'

    if os.path.isfile(jsonPath):
        jsonData = json.load(open(jsonPath))
    else:
        jsonData = []

    # Optional rescaling
    if rescaling == 'off' or rescaling == 'auto':
        pass

    if rescaling == 'basic':
        image, info = rescale(image, info)

    # NOTE: Other approaches to rescaling and / or converting from raw file values could be added (including cases
    # where the json sidecar is unavailable)
    if ('Manufacturer' in jsonData) and (jsonData['Manufacturer'] == 'Siemens') \
            and (check_json_image_type(jsonData) == 'phase'):

        image, info = convert_siemens_phase(image, info)

    else:
        image, info = rescale(image, info)

    return image, info, jsonData

# -----------------------------------------------------------------------------
# Local functions
# -----------------------------------------------------------------------------


def convert_siemens_phase(img, info):

    PHASE_SCALING_SIEMENS = 4096

    if (info.header['scl_inter'] == -PHASE_SCALING_SIEMENS) and (info.header['scl_slope'] == 2):

        img, infoConverted = rescale(img, info)
        imgConverted = img * (math.pi / PHASE_SCALING_SIEMENS)

        # Update header: 16 is the NIfTI code for float; bitpix = number of bits
        infoConverted.header['datatype'] = 16
        infoConverted.header['bitpix'] = 32

    else:
        print('Warning: The nii header differs from that expected of Siemens phase data.\n '
              'Output values (units) are effectively unknown')
        imgConverted, infoConverted = rescale(img, info)
    return imgConverted, infoConverted


def check_json_image_type(jsonData):
    imgType = []

    if not jsonData:
        return imgType

    assert isinstance(jsonData, dict)

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


def rescale(img, info):
    img1 = info.AdditiveOffset + np.multiply(info.MultiplicativeScaling, img)

    # Check for possible integer overflow / type issues
    if np.array_equal(img1, float(info.AdditiveOffset) + float(info.MultiplicativeScaling) * float(img)):

        imgRescaled = img1
        infoRescaled = info

        # Update header:
        infoRescaled.MultiplicativeScaling = 1
        infoRescaled.AdditiveOffset = 0
        infoRescaled.raw.scl_slope = 1
        infoRescaled.raw.scl_inter = 0

    else:
        print('Warning: Aborting image rescaling to avoid integer overflow.\nThe NIfTI header may contain errors.\n')

        imgRescaled = img
        infoRescaled = info

    return imgRescaled, infoRescaled
