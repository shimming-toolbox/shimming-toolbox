#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy as np
import nibabel as nib
import json
import math

logger = logging.getLogger(__name__)
PHASE_SCALING_SIEMENS = 4096

def load_nifti(path_data):
    """
    Load data from a NIFTI type file with dcm2bids.
    Args:
        path_data (str): absolute or relative path to the directory the acquisition data
    Returns:
        info (Nifti1Image.Header): List containing all information from every Nifti image
        json_info (dict): List containing all information in JSON format from every Nifti image
        niftis (ndarray): 5D array of all acquisition in time (x, y, z, echo, volume)

    Note:
        If 'path' is a folder containing niftis, directly output niftis. It 'path' is a folder containing acquisitions,
        ask the user for which acquisition to use.
    """
    if not os.path.exists(path_data):
        raise RuntimeError("Not an existing NIFTI path")

    file_list = []
    [file_list.append(os.path.join(path_data, f)) for f in os.listdir(path_data) if f not in file_list]

    nifti_path = ""
    if all([os.path.isdir(f) for f in file_list]):
        acquisitions = [f for f in file_list if os.path.isdir(f)]
        logging.info("Multiple acquisition directories in path. Choosing only one.")
    elif all([os.path.isfile(f) for f in file_list]):
        logging.info("Acqusition directory given. Using acquisitions.")
        nifti_path = path_data
    else:
        raise RuntimeError("Directories and files in input path")


    if not nifti_path:
        for i in range(len(acquisitions)):
            logging.info("{}:{}\n".format( i, os.path.basename(file_list[i])))

        select_acquisition = -1
        while 1:
            input_resp = input("Enter the number for the appropriate acquisition folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_acquisition = int(input_resp)

            if (select_acquisition in range(len(acquisitions))):
                break
            else:
                logging.error("Input must be linked to an acquisition folder. {} is out of range".format(input_resp))

        nifti_path = os.path.abspath(file_list[select_acquisition])

    nifti_list = [os.path.join(nifti_path, f) for f in os.listdir(nifti_path) if (f.endswith(".nii") or f.endswith(".nii.gz"))]
    n_echos = len(nifti_list)

    if n_echos <= 0:
        raise RuntimeError("No acquisition images in selected path {}".format(nifti_path))

    info_init, json_init, img_init = read_nii(nifti_list[0])

    info = []
    json_info = []
    niftis = np.empty([1, 1], dtype=float)
    if info_init.ndim == 3:
        niftis = np.empty([info_init.shape[0], info_init.shape[1], info_init.shape[2], n_echos, 1], dtype = float)
        for i_echo in range(n_echos):
            tmp_nii = read_nii(os.path.abspath(nifti_list[i_echo]))
            info.append(tmp_nii[0].header)
            json_info.append(tmp_nii[1])
            niftis[:, :, :, i_echo, 0] = tmp_nii[2]
    else:
        niftis = np.empty([info_init.shape[0], info_init.shape[1], info_init.shape[2], n_echos, info_init.shape[3], info_init.shape[3]], dtype=float)
        for i_echo in range(n_echos):
            tmp_nii = read_nii(os.path.abspath(nifti_list[i_echo]))
            info.append(tmp_nii[0].header)
            json_info.append(tmp_nii[1])
            for i_volume in range(info_init.shape[3]):
                niftis[:, :, :, i_echo, i_volume] = tmp_nii[2][:, :, :, i_volume]

    return niftis, info, json_info


def read_nii(nii_path, auto_scale = True):
    """ Reads a nifti file and returns the corresponding image and info. Also returns the associated json data.
    Args:
        nii_path (str): direct path to the .nii or .nii.gz file that is going to be read
        auto_scale (:obj:`bool`, optional): Tells if scaling is done before return
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

    image = np.asanyarray(info.dataobj)
    if auto_scale:
        if ('Manufacturer' in json_data) and (json_data['Manufacturer'] == 'Siemens') and (image_type(json_data) == 'phase'):
            image = image * (2 * math.pi / PHASE_SCALING_SIEMENS)

    return info, json_data, image


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


if __name__ == "__main__":
    load_nifti("C:\\Users\\Gabriel\\Documents\\share\\test_nifti\\sub-")