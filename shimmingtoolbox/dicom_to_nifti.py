# coding: utf-8

# TODO: check in unit test if dcm2bids_scaffold is installed, and also check for the required version.

import subprocess

from shimmingtoolbox import __dir_config_dcm2bids__


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder

    """
    cmd = 'dcm2bids -d {} -p {} -o {} -l DEBUG -c {}'.format(
        path_dicom, subject_id, path_nifti, path_config_dcm2bids)
    subprocess.run(cmd.split(' '))
