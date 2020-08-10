# coding: utf-8

import json
import nibabel as nib
import numpy as np
import os
import subprocess
import shutil

from shimmingtoolbox import __dir_config_dcm2bids__


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', path_config_dcm2bids=__dir_config_dcm2bids__,
                   special_dicom=''):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): path to the input dicom folder
        path_nifti (str): path to the output nifti folder
        subject_id (str): ID of subject
        path_config_dcm2bids (str): Path to dcm2bids.json file
        special_dicom (str): Name of the modality for which the output NIFTI from dcm2niix should
        be half-split along the last dimension. Available values: 'TB1map'.
    """

    cmd = f'dcm2bids -d {path_dicom} -p {subject_id} -o {path_nifti} -l DEBUG -c {path_config_dcm2bids}'
    subprocess.run(cmd.split(' '))

    # If output nifti needs to be split along the last dimension. This could happen if there is not enough information
    # in the input dicom data.
    if special_dicom != '':
        list_special_dicom = ['TB1map']
        assert special_dicom in list_special_dicom, f'The specified special_dicom name is invalid.' \
                                                    f' Available names: {list_special_dicom}'
        path_complete = os.path.join(path_nifti, subject_id, special_dicom, f'{subject_id}_{special_dicom}')
        nifti = nib.load(path_complete + '.nii.gz')
        nifti_image = nifti.get_fdata()
        n_coils = nifti_image.shape[3]
        image_mag = nib.Nifti2Image(nifti_image[:, :, :, :int(n_coils/2)], np.eye(4))
        image_phase = nib.Nifti2Image(nifti_image[:, :, :, int(n_coils/2):], np.eye(4))
        with open(path_complete + '.json') as json_file:
            json_mag = json_phase = json.load(json_file)
        json_mag['ImageComments'] = 'magnitude'
        json_phase['ImageComments'] = 'phase'
        # Create the magnitude NIfTI
        nib.save(image_mag, path_complete + '_mag.nii.gz')
        # Create json sidecar for the magnitude NIfTI
        with open(path_complete + '_mag.json', 'w') as json_mag_file:
            json.dump(json_mag, json_mag_file, indent=4, sort_keys=True)
        # Create the phase NIfTI
        nib.save(image_phase, path_complete + '_phase.nii.gz')
        # Create json sidecar for the phase NIfTI
        with open(path_complete + '_phase.json', 'w') as json_phase_file:
            json.dump(json_phase, json_phase_file, indent=4, sort_keys=True)
        # Delete original json file
        os.remove(path_complete + '.json')
        # Delete original nifti file
        os.remove(path_complete + '.nii.gz')
