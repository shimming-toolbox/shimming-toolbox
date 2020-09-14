#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy as np
import nibabel as nib
import json
import math

from bids import BIDSLayout

from shimmingtoolbox import __dir_config_pybids__

logger = logging.getLogger(__name__)
PHASE_SCALING_SIEMENS = 4096


def load_nifti(path_data, modality='phase', bids_config = __dir_config_pybids__):
    """
    Load data from a directory containing NIFTI type file with nibabel.
    Args:
        path_data (str): Path to the directory containing the file(s) to load
        modality (str): Modality to read nifti (can be phase or magnitude)
    Returns:
        nibabel.Nifti1Image.Header: List containing headers for every Nifti file
        dict: List containing all information in JSON format from every Nifti image
        numpy.ndarray: 5D array of all acquisition in time (x, y, z, echo, volume)

    Note:
        If 'path' is a folder containing niftis, directly output niftis. It 'path' is a folder containing acquisitions,
        ask the user for which acquisition to use.
    """
    if not os.path.exists(path_data):
        raise RuntimeError("Not an existing NIFTI path")
    if not os.path.exists(bids_config):
        bids_config = None

    bids_layout = BIDSLayout(root=path_data, validate=False, config=bids_config)

    sub_list = bids_layout.get(return_type='id', target='subject')
    select_sub = 0
    if sub_list:
        for i in range(len(sub_list)):
            print(f"{i}:{sub_list[i]}\n")
        while 1:
            input_resp = input("Enter the number for the appropriate subject folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_sub = int(input_resp)

            if select_sub in range(len(sub_list)):
                break
            else:
                logging.error(f"Input must be linked to a subject folder. {input_resp} is out of range")

    ses_list = bids_layout.get(return_type='id', target='session', subject=sub_list[select_sub])
    select_ses = 0
    if ses_list:
        for i in range(len(ses_list)):
            print(f"{i}:{ses_list[i]}\n")
        while 1:
            input_resp = input("Enter the number for the appropriate subject folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_ses = int(input_resp)

            if select_ses in range(len(ses_list)):
                break
            else:
                logging.error(f"Input must be linked to a subject folder. {input_resp} is out of range")

    datatype_list = bids_layout.get(return_type='id', target='session', subject=sub_list[select_sub],
                                    session=ses_list[select_ses])
    select_datatype = 0
    if datatype_list:
        for i in range(len(datatype_list)):
            print(f"{i}:{datatype_list[i]}\n")
        while 1:
            input_resp = input("Enter the number for the appropriate subject folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_datatype = int(input_resp)

            if select_datatype in range(len(datatype_list)):
                break
            else:
                logging.error(f"Input must be linked to a subject folder. {input_resp} is out of range")

    acq_list = bids_layout.get(return_type='id', target='session', subject=sub_list[select_sub],
                                    session=ses_list[select_ses], datatype=datatype_list[select_datatype])
    select_acq = 0
    if acq_list:
        for i in range(len(acq_list)):
            print(f"{i}:{acq_list[i]}\n")
        while 1:
            input_resp = input("Enter the number for the appropriate subject folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_acq = int(input_resp)

            if select_acq in range(len(acq_list)):
                break
            else:
                logging.error(f"Input must be linked to a subject folder. {input_resp} is out of range")

    image_file_list = bids_layout.get(subject=sub_list[select_sub], session=ses_list[select_ses], acquisition=acq_list[select_acq],
                                      datatype=datatype_list[select_datatype], return_type='filename', extension=['nii.gz', 'nii'])

    info = []
    json_info = []
    nifti_init = [read_nii(image_file_list[i]) for i in range(len(image_file_list))]
    echo_shape = sum(1 for tmp_info in nifti_init if modality in tmp_info[1]['ImageComments'])
    niftis = np.empty([nifti_init[0][0].shape[0], nifti_init[0][0].shape[1], nifti_init[0][0].shape[2], echo_shape,
                       (1 if nifti_init[0][0].ndim == 3 else nifti_init[0][0].shape[3])], dtype=float)
    nifti_pos = 0
    for file_path in image_file_list:
        tmp_nii = read_nii(file_path)
        if modality in tmp_nii[1]['ImageComments']:
            info.append(tmp_nii[0].header)
            json_info.append(tmp_nii[1])
            if niftis.shape[4] == 1:
                niftis[:, :, :, nifti_pos, 0] = tmp_nii[2]
            else:
                for i_volume in range(nifti_init[0][0].shape[3]):
                    niftis[:, :, :, nifti_pos, i_volume] = tmp_nii[2][:, :, :, i_volume]
            nifti_pos += 1

    return niftis, info, json_info


def read_nii(nii_path, auto_scale=True):
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

    image = np.asarray(info.dataobj)
    if auto_scale:
        if ('Manufacturer' in json_data) and (json_data['Manufacturer'] == 'Siemens') \
                and (("*phase*" in json_data['ImageComments']) or ("P" in json_data["ImageType"])):
            image = image * (2 * math.pi / PHASE_SCALING_SIEMENS)

    return info, json_data, image