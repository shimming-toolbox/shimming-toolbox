#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy

logger = logging.getLogger(__name__)

def load_nifti(file_path):
    """
    Load data from a NIFTI type file with dcm2bids.
    Args:
        file_path (str): absolute or relative path to the directory the acquisition data
    Returns:
        info (ndarray): List containing all information from every Nifti image
        json_info (ndarray): List containing all information in JSON format from every Nifti image
        niftis (ndarray): Array of all acquisition in time
    """

    if not os.path.exists(file_path):
        raise("Not an existing NIFTI path")

    file_list = []
    [file_list.append(os.path.join(file_path,f)) for f in os.listdir(file_path) if f not in file_list]

    nifti_path = ""
    if all([os.path.isdir(f) for f in file_list]):
        acquisitions = [f for f in file_list if os.path.isdir(f)]
        print("Multiple acquisition directories in path. Choosing only one.")
    elif all([os.path.isfile(f) for f in file_list]):
        print("Acqusition directory given. Using acquisitions.")
        nifti_path = file_path
    else:
        raise ("Directories and files in input path")


    if not nifti_path:
        for i in range(len(acquisitions)):
            print("{}:{}\n".format( i, os.path.basename(file_list[i])))

        select_acquisition = -1
        while 1:
            input_resp = input("Enter the number for the appropriate acquisition folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_acquisition = int(input_resp)

            if (select_acquisition in range(len(acquisitions))):
                print("Input must be linked to an acquisition folder. {} is out of range".format(input_resp))
                break

        nifti_path = os.path.abspath(file_list[select_acquisition])

    nifti_list = [f for f in os.listdir(os.path.abspath(nifti_path)) if (f.endswith(".nii") or f.endswith(".nii.gz"))]
    n_echos = len(nifti_list)

    if n_echos <= 0:
        raise("No acquisition images in selected path {}".format(nifti_path))

    _, info_init, _ = read_nii(nifti_list[1])

    niftis = numpy.empty([info_init.x, info_init.y, info_init.z, n_echos, info_init.time], dtype = float)
    info = numpy.empty([n_echos], dtype = int)
    json_info = numpy.empty([n_echos], dtype = str)

    if len(info.ImageSize) == 3:
        for i_echo in range(n_echos):
            info[i_echo], json_info[i_echo],  niftis[:,:,:,i_echo,:] = read_nii(os.path.abspath(nifti_list[i_echo]))
    else:
        for i_echo in range(n_echos):
            info[i_echo], json_info[i_echo], niftis[:, :, :, i_echo, :] = read_nii(os.path.abspath(nifti_list[i_echo]))



if __name__ == "__main__":
    load_nifti("C:\\Users\\Gabriel\\Documents\\share\\test_nifti\\sub-")