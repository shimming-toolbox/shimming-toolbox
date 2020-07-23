#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy

logger = logging.getLogger(__name__)

def load_nifti(file_path):
    """
    Load data from a NIFTI type file with dcm2bids.
    :param file_path: file path for the data
    :return: nifti data, info and json_info
    """

    if not os.path.exists(file_path):
        raise("Not an existing NIFTI path")

    file_list = []
    [file_list.append(f) for f in os.listdir(file_path) if f not in file_list]

    if len([f for f in file_list if os.path.isdir(f)]) != 1:
        raise("Directories and files in input path")
    else:
        acquisitions = [f for f in file_list if os.path.isdir(f)]

    nifti_path = ""
    if acquisitions:
        for i in range(len(acquisitions)):
            print("{}:{}\n".format( i, file_list[i]))

        select_acquisition = -1
        while 1:
            input_resp = input("Enter the number for the appropriate acquisition folder, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0

            select_acquisition = int(input_resp)

            if (select_acquisition in range(len(acquisitions))):
                break

        nifti_path = os.path.abspath(file_list[select_acquisition])
    else:
        nifti_path = file_path

    nifti_list = [f for f in os.listdir(os.path.abspath(nifti_path)) if f.endswith(".nii")]
    n_echos = len(nifti_list)

    if n_echos <= 0:
        raise("No acquisition images in selected path {}".format(nifti_path))

    _, info_init, _ = imutils.read_nii(nifti_list[1])

    niftis = numpy.empty([info_init.x, info_init.y, info_init.z, n_echos, info_init.time], dtype = float)
    info = numpy.empty([n_echos], dtype = int)
    json_info = numpy.empty([n_echos], dtype = str)

    if len(info.ImageSize) == 3:
        for i_echo in range(n_echos):
            niftis[:,:,:,i_echo,:], info[i_echo], json_info[i_echo] = imutils.read_nii(os.path.abspath(nifti_list[i_echo]))
    else
        for i_echo in range(n_echos):
            niftis[:, :, :, i_echo, :], info[i_echo], json_info[i_echo] = imutils.read_nii(os.path.abspath(nifti_list[i_echo]))

if __name__ == "__main__":
    load_nifti("..")