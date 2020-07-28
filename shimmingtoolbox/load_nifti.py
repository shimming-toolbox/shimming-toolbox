#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy

from shimmingtoolbox.read_nii import read_nii

logger = logging.getLogger(__name__)

def load_nifti(file_path):
    """
    Load data from a NIFTI type file with dcm2bids.
    :param file_path: absolute or relative path to the directory the acquisition data
    :return:    List containing all information from every Nifti image
                List containing all information in JSON format from every Nifti image
                5D array of all acquisition in time (x, y, z, time, echoe)

    Note:
        If 'path' is a folder containing niftis, directly output niftis. It 'path' is a folder containing acquisitions,
        ask the user for which acquisition to use.
    """

    if not os.path.exists(file_path):
        raise("Not an existing NIFTI path")

    file_list = []
    [file_list.append(os.path.join(file_path, f)) for f in os.listdir(file_path) if f not in file_list]

    nifti_path = ""
    if all([os.path.isdir(f) for f in file_list]):
        acquisitions = [f for f in file_list if os.path.isdir(f)]
        print("Multiple acquisition directories in path. Choosing only one.")
    elif all([os.path.isfile(f) for f in file_list]):
        #TODO check if JSON available
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
                break
            else:
                print("Input must be linked to an acquisition folder. {} is out of range".format(input_resp))

        nifti_path = os.path.abspath(file_list[select_acquisition])

    nifti_list = [os.path.join(nifti_path, f) for f in os.listdir(nifti_path) if (f.endswith(".nii") or f.endswith(".nii.gz"))]
    n_echos = len(nifti_list)

    if n_echos <= 0:
        raise("No acquisition images in selected path {}".format(nifti_path))

    _, _, img_init = read_nii(nifti_list[0])


    niftis = numpy.empty([img_init.shape[0], img_init.shape[1], img_init.shape[2], n_echos], dtype = float)
    info = []
    json_info = []

    for i_echo in range(n_echos):
        #TODO Check read_nii
        tmp_nii = read_nii(os.path.abspath(nifti_list[i_echo]))
        info.append(tmp_nii[0].header)
        json_info.append(tmp_nii[1])
        niftis[:, :, :, i_echo, :] = tmp_nii[2]

    return niftis, info, json_info


if __name__ == "__main__":
    load_nifti("C:\\Users\\Gabriel\\Documents\\share\\test_nifti\\sub-")