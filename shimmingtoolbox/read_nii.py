import json
import math
import nibabel as nib
import os

def read_nii(nii_path):
    """
    Args:
        nii_path (str): direct path to the .nii or .nii.gz file

    Returns:

    """

    info = nib.load(nii_path)
    image = info.get_fdata()

    # `extractBefore` should get the correct filename in both.nii and.nii.gz cases
    json_path = nii_path.split('.nii')[0] + '.json'

    if os.path.isfile(json_path):
        json_data = json.load(open(json_path))
    else:
        raise ValueError('Ambiguous ImageType entry in json file: Indicates magnitude AND phase?')

    # NOTE: nib.load automatically scales the nifti and replaces scl_inter and scl_slope with 'nan'
    # More info in the "Data scaling" section in https://nipy.org/nibabel/nifti_images.html
    if ('Manufacturer' in json_data) and (json_data['Manufacturer'] == 'Siemens') \
            and (check_json_image_type(json_data) == 'phase'):

        image, info = convert_siemens_phase(image, info)

    return image, info, json_data


def check_json_image_type(json_data):
    img_type = []

    # Check that jsonData exists
    if not json_data:
        raise TypeError("json_data is empty")

    # Check that jsonData is a dictionary
    if not isinstance(json_data, dict):
        raise TypeError("json_data is not a dictionary")

    if 'ImageType' in json_data:
        is_phase = "P" in json_data['ImageType']
        is_mag = "M" in json_data['ImageType']

        if is_phase and is_mag:
            # Both true: json file and/or DICOM issue
            raise ValueError('Ambiguous ImageType entry in json file: Indicates magnitude AND phase')
        elif is_phase:
            img_type = 'phase'
        elif is_mag:
            img_type = 'magnitude'
        else:
            if ('Manufacturer' in json_data) and (json_data['Manufacturer'] != 'Siemens'):
                raise ValueError('Unknown image type. Possibly due to images sourced from non-Siemens MRI')
            else:
                raise ValueError('Unknown image type')

        return img_type


def convert_siemens_phase(image, info):

    PHASE_SCALING_SIEMENS: float = 4096

    info_converted = info
    img_converted = image * (math.pi / PHASE_SCALING_SIEMENS)

    # Update header: 16 is the NIfTI code for float; bitpix = number of bits
    info_converted.header['datatype'] = 16
    info_converted.header['bitpix'] = 32

    return img_converted, info_converted
