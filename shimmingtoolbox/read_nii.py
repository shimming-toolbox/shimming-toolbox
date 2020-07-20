import json
import math
import nibabel as nib
import numpy as np
import os


def read_nii(nii_path, return_image=False):
    """ Reads a nifti file and returns the corresponding image and info. Also returns the associated json data.

    Args:
        nii_path (str): direct path to the .nii or .nii.gz file that is going to be read
        return_image (:obj:`bool`, optional): Defines if the function should return `image`

    Returns:
        info (Nifti1Image): Objet containing various data about the nifti file (returned by nibabel.load)
        json_data (dict): Contains the different fields present in the json file corresponding to the nifti file
        image (ndarray): Image contained in the read nifti file. Siemens phase images are rescaled between 0 and 2pi.

    """

    info = nib.load(nii_path)

    # `extractBefore` should get the correct filename in both.nii and.nii.gz cases
    json_path = nii_path.split('.nii')[0] + '.json'

    if os.path.isfile(json_path):
        json_data = json.load(open(json_path))
    else:
        raise ValueError('Missing json file')

    if return_image:
        image = np.asanyarray(info.dataobj)
        # NOTE: nib.load automatically scales the nifti and replaces scl_inter and scl_slope with 'nan' in the header
        # More info in the "Data scaling" section in https://nipy.org/nibabel/nifti_images.html
        if ('Manufacturer' in json_data) and (json_data['Manufacturer'] == 'Siemens') \
                and (image_type(json_data) == 'phase'):

            image = rescale_siemens_phase(image)
        return info, json_data, image
    else:
        return info, json_data


def image_type(json_data):
    """ Returns the nifti image type indicated by the json file

    Args:
        json_data (dict): Contains the same fields as the json file corresponding to a nifti file

    Returns:
        img_type (str): Type of the image. It can take the values `phase`, `magnitude`.

    """

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


def rescale_siemens_phase(image):
    """ Rescales a siemens phase image between 0 and 2*pi

    Args:
        image (ndarray): Phase image from a siemens system with integer values from 0 to 4095

    Returns:
        img_rescaled (ndarray): Rescaled phase image with float values from -pi to pi

    """

    PHASE_SCALING_SIEMENS = 4096

    img_rescaled = image * (2*math.pi / PHASE_SCALING_SIEMENS)

    return img_rescaled
