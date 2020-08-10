# coding: utf-8

import os
import subprocess
import shutil
import tempfile

from shimmingtoolbox import __dir_config_dcm2bids__


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__,
                   special_dicom=[]):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder
        subject_id (str): ID of subject
        path_config_dcm2bids (str): Path to dcm2bids.json file
        special_dicom (:obj:`list` of :obj:`str`): List of modalities for which the output NIFTI from dcm2niix should
        be half-split along the last dimension.
    """
    # # Case where the DICOM files have
    # if special_dicom:
    #     with tempfile.TemporaryDirectory(prefix='st_mag_') as tmp_mag:
    #         dicom_number = len(os.listdir(path_dicom))
    #         dicom_list = os.listdir(path_dicom)
    #         for i in range(int(dicom_number/2)):
    #             shutil.copy(os.path.join(path_dicom, dicom_list[i]), tmp_mag)
    #         cmd = f'dcm2bids -d {tmp_mag} -p {subject_id} -o {path_nifti} -l DEBUG -c {path_config_dcm2bids}'
    #         subprocess.run(cmd.split(' '))
    #
    #     with tempfile.TemporaryDirectory(prefix='st_mag_') as tmp_phase:
    #         dicom_number = len(os.listdir(path_dicom))
    #         dicom_list = os.listdir(path_dicom)
    #         for i in range(int(dicom_number/2), int(dicom_number)):
    #             shutil.copy(os.path.join(path_dicom, dicom_list[i]), tmp_phase)
    #         cmd = f'dcm2bids -d {tmp_phase} -p {subject_id} -o {path_nifti} -l DEBUG -c {path_config_dcm2bids}'
    #         subprocess.run(cmd.split(' '))
    #
    # else:
    cmd = f'dcm2bids -d {path_dicom} -p {subject_id} -o {path_nifti} -l DEBUG -c {path_config_dcm2bids}'
    subprocess.run(cmd.split(' '))

    # If output nifti needs to be split along the last dimension. This could happen if there is not enough information
    # in the input dicom data.
    # TODO: assert input special_dicom
    if special_dicom is not []:
        a=1