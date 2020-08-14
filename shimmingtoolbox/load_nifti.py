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


def load_nifti(path_data, modality = 'phase'):
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
        logging.info("Acquisition directory given. Using acquisitions.")
        nifti_path = path_data
    else:
        raise RuntimeError("Directories and files in input path")

    if not nifti_path:
        for i in range(len(acquisitions)):
            print(f"{i}:{os.path.basename(file_list[i])}\n")

        select_acquisition = -1
        while 1:
            input_resp = input("Enter the number for the appropriate acquisition folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_acquisition = int(input_resp)

            if select_acquisition in range(len(acquisitions)):
                break
            else:
                logging.error(f"Input must be linked to an acquisition folder. {input_resp} is out of range")

        nifti_path = os.path.abspath(file_list[select_acquisition])

    nifti_list = [os.path.join(nifti_path, f) for f in os.listdir(nifti_path) if f.endswith((".nii", ".nii.gz"))]

    info_init = [read_nii(nifti_list[i]) for i in range(len(nifti_list))]

    info = []
    json_info = []
    niftis = np.empty([1, 1], dtype=float)

    echo_list = []
    run_list = {}
    n_echos = 0
    for file_info in info_init:
        if (file_info[1]['AcquisitionNumber'] not in run_list.keys()):
            run_list[file_info[1]['AcquisitionNumber']] = []
        if (file_info[1]['EchoNumber'] not in echo_list):
            n_echos += 1
            echo_list.append(file_info[1]['EchoNumber'])
        run_list[file_info[1]['AcquisitionNumber']].append((file_info, file_info[1]['ImageComments']))

    select_run = -1
    if len(list(run_list.keys())) > 1:
        for i in list(run_list.keys()):
            logging.info(f"{i}\n")

        while 1:
            input_resp = input("Enter the number for the appropriate run number, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_run = int(input_resp)

            if select_run in range(len(run_list.keys())):
                break
            else:
                logging.error(f"Input must be linked to a run number. {input_resp} is out of range")
    else:
        select_run = list(run_list.keys())[0]
        logging.info(f"Reading acquisitions for run {list(run_list.keys())[0]}")

    nifti_pos = 0
    echo_shape = sum(1 for tmp_info in run_list[select_run] if modality in tmp_info[1])
    if info_init[0][0].ndim == 3:
        niftis = np.empty([info_init[0][0].shape[0], info_init[0][0].shape[1], info_init[0][0].shape[2], echo_shape, 1], dtype=float)
        for i_echo in range(n_echos):
            tmp_nii = run_list[select_run][i_echo][0]
            if modality in run_list[select_run][i_echo][1]:
                info.append(tmp_nii[0].header)
                json_info.append(tmp_nii[1])
                niftis[:, :, :, nifti_pos, 0] = tmp_nii[2]
                nifti_pos += 1
    else:
        niftis = np.empty([
            info_init[0][0].shape[0], info_init[0][0].shape[1], info_init[0][0].shape[2],
            echo_shape, info_init[0][0].shape[3]], dtype=float)
        for i_echo in range(n_echos):
            if modality in run_list[select_run][i_echo][1]:
                tmp_nii = run_list[select_run][i_echo][0]
                info.append(tmp_nii[0].header)
                json_info.append(tmp_nii[1])
                for i_volume in range(info_init[0][0].shape[3]):
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
                and ("P" in json_data['ImageType']):
            image = image * (2 * math.pi / PHASE_SCALING_SIEMENS)

    return info, json_data, image

if __name__ == "__main__":
    load_nifti("C:\\Users\\Gabriel\\Documents\\share\\008_a_gre_DYNshim1")
