#!usr/bin/env python3
# -*- coding: utf-8

from shutil import copytree
import json
import logging
import nibabel as nib
import numpy as np
import os
from dcm2bids.dcm2bids_gen import Dcm2BidsGen
from dcm2bids.utils.tools import check_latest
from dcm2bids import version
import shutil

from shimmingtoolbox import __config_dcm2bids__
from shimmingtoolbox.coils.coordinates import get_main_orientation
from shimmingtoolbox.utils import create_output_dir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GAMMA = 2.675e8  # Proton's gyromagnetic ratio (rad/(T.s))
SATURATION_FA = 90  # Saturation flip angle hard-coded in TFL B1 mapping sequence (deg)


def dicom_to_nifti(path_dicom, path_nifti, subject_id='sub-01', fname_config_dcm2bids=__config_dcm2bids__,
                   remove_tmp=False):
    """ Converts dicom files into nifti files by calling dcm2bids

    Args:
        path_dicom (str): Path to the input DICOM folder.
        path_nifti (str): Path to the output NIfTI folder.
        subject_id (str): Name of the imaged subject.
        fname_config_dcm2bids (str): Path to the dcm2bids config JSON file.
        remove_tmp (bool): If True, removes the tmp folder containing the NIfTI files created by dcm2niix.
    """

    # Create the folder where the nifti files will be stored
    if not os.path.exists(path_dicom):
        raise FileNotFoundError("No dicom path found")
    if not os.path.exists(fname_config_dcm2bids):
        raise FileNotFoundError("No dcm2bids config file found")

    if os.path.isfile(path_nifti):
        raise ValueError("Output NIfTI path should be a folder")

    create_output_dir(path_nifti)

    # Create bids structure for data
    logger.info(f"dcm2bids version: {version.__version__}")

    # Create derivatives folder
    path_derivatives = os.path.join(path_nifti, 'derivatives')
    if not os.path.exists(path_derivatives):
        os.makedirs(path_derivatives)

    # Copy original dicom files into nifti_path/sourcedata
    copytree(path_dicom, os.path.join(path_nifti, 'sourcedata'), dirs_exist_ok=True)

    # Update the PATH environment variable to include the dcm2niix executable
    # TODO: Try this out on Windows, path could be Scripts instead of bin
    if 'ST_DIR' in os.environ and os.path.exists(os.environ['ST_DIR']):
        os.environ['PATH'] = os.path.join(os.environ['ST_DIR'], 'python', 'bin') + os.pathsep + os.environ['PATH']
    else:
        logger.warning("Environment variable ST_DIR not found. Using default path.")

    # Run dcm2bids
    check_latest('dcm2bids')
    Dcm2BidsGen(path_dicom, subject_id, fname_config_dcm2bids, path_nifti).run()

    # In the special case where a phasediff should be created but the filename is phase instead. Find the file and
    # rename it
    rename_phasediff(path_nifti, subject_id)

    # Go in the RF map folder
    path_rfmap = os.path.join(path_nifti, 'sub-' + subject_id, 'rfmap')
    if os.path.exists(path_rfmap):
        # Make a list of the json files in rfmap folder
        file_list = []
        [file_list.append(os.path.join(path_rfmap, f)) for f in os.listdir(path_rfmap) if
         os.path.splitext(f)[1] == '.json']
        file_list = sorted(file_list)
        for fname_json_b1 in file_list:
            # Do nothing if the files have already been processed by Shimming-Toolbox
            if '_uncombined' in fname_json_b1 or '_shimmed' in fname_json_b1:
                continue
            # Open the json file
            with open(fname_json_b1) as json_file:
                json_data = json.load(json_file)
                # Check what B1+ mapping sequence has been used to proceed accordingly
                # If Siemens' TurboFLASH B1 mapping (dcm2niix cannot separate phase and magnitude for this sequence)
                if ('SequenceName' in json_data) and 'tfl2d1_16' in json_data['SequenceName']:
                    fname_nii_b1 = fname_json_b1.split(".json")[0] + ".nii.gz"
                    nii_b1 = nib.load(fname_nii_b1)
                    nii_b1_new = fix_tfl_b1(nii_b1, json_data)

                    # Save uncombined B1+ maps in a NIfTI file that can now be visualized in FSLeyes
                    json_data["ImageComments"] = 'Complex uncombined B1+ map (nT/V)'
                    fname_nii_b1_new = fname_nii_b1.split('.nii')[0] + '_uncombined.nii' + fname_nii_b1.split('.nii')[1]
                    nib.save(nii_b1_new, fname_nii_b1_new)

                    # Save the associated JSON file
                    fname_json_b1_new = open(os.path.join(fname_nii_b1_new.split('.nii')[0] + '.json'), mode='w')
                    json.dump(json_data, fname_json_b1_new)

                    # Remove the old buggy NIfTI and associated JSON files
                    os.remove(fname_nii_b1)
                    os.remove(fname_json_b1)
                # TODO: Add handling of other B1+ mapping sequences

        logger.info("B1+ NIfTI have been reshuffled and rescaled.")

    if remove_tmp:
        shutil.rmtree(os.path.join(path_nifti, 'tmp_dcm2bids'))


def rename_phasediff(path_nifti, subject_id):
    # Dcm2bids removes 'sub-' if it is in the subject name, otherwise it would be there twice
    subject_id = subject_id.split('sub-', maxsplit=1)[-1]

    path_fmap = os.path.join(path_nifti, f"sub-{subject_id}", 'fmap')
    if os.path.exists(path_fmap):
        # Make a list of the json files in fmap folder
        file_list = []
        [file_list.append(os.path.join(path_fmap, f)) for f in os.listdir(path_fmap)
         if os.path.splitext(f)[1] == '.json']
        file_list = sorted(file_list)

        for fname_json in file_list:
            is_renaming = False
            # Open the json file
            with open(fname_json) as json_file:
                json_data = json.load(json_file)
                # Make sure it is phase data and that the keys EchoTime1 and EchoTime2 are defined and that
                # the tag "sequenceName" includes fm2d2 which is Siemens' sequence that outputs a phasediff
                if ('ImageType' in json_data) and ('P' in json_data['ImageType']) and \
                   ('EchoTime1' in json_data) and ('EchoTime2' in json_data) and \
                   ('SequenceName' in json_data) and ('fm2d2' in json_data['SequenceName']) and \
                   ('EchoNumber' in json_data) and (int(json_data['EchoNumber']) == 2):
                    # Make sure it is not already named phasediff
                    if len(os.path.basename(fname_json).split(subject_id, 1)[-1].rsplit('phasediff', 1)) == 1:
                        # Split the filename in 2 and remove phase
                        file_parts = fname_json.rsplit('phase', 1)

                        # EchoTime1 and EchoTime2 are written even if it's a dual echo (not a phasediff), this makes
                        # sure that if echo 1 exists, that we do not rename echo2 to phasediff
                        if os.path.isfile(file_parts[0] + "phase1.json"):
                            continue

                        if len(file_parts) == 2:
                            # Stitch the filename back together making sure to remove any digits that could be after
                            # 'phase'
                            digits = '0123456789'
                            fname_new_json = file_parts[0] + 'phasediff' + file_parts[1].lstrip(digits)
                            is_renaming = True

            # Rename the json and nifti file
            if is_renaming:
                if os.path.exists(os.path.splitext(fname_json)[0] + '.nii.gz'):
                    fname_nifti_new = os.path.splitext(fname_new_json)[0] + '.nii.gz'
                    fname_nifti_old = os.path.splitext(fname_json)[0] + '.nii.gz'
                    logger.debug(f"Renaming file: {fname_nifti_old} to: {fname_nifti_new}")
                    os.rename(fname_nifti_old, fname_nifti_new)
                    os.rename(fname_json, fname_new_json)


def fix_tfl_b1(nii_b1, json_data):
    """Un-shuffles and rescales the magnitude and phase of complex B1+ maps acquired with Siemens' standard B1+ mapping
    sequence. Also computes a corrected affine matrix allowing the B1+ maps to be visualized in FSLeyes.
    Args:
        nii_b1 (numpy.ndarray): Array of dimension (x, y, n_slices, 2*n_channels) as created by dcm2niix.
        json_data (dict): Contains the different fields present in the json file corresponding to the nifti file.

    Returns:
        nib.Nifti1Image: NIfTI object containing the complex rescaled B1+ maps (x, y, n_slices, n_channels).
    """
    image = np.array(nii_b1.dataobj)
    # The number of slices corresponds to the 3rd dimension of the shuffled NIfTI volume.
    n_slices = image.shape[2]
    # The number of Tx channels corresponds to the 4th dimension of the shuffled NIfTI of the shuffled NIfTI volume.
    n_channels = image.shape[3]//2

    # Magnitude values are stored in the first half of the 4th dimension
    b1_mag = image[:, :, :, :n_channels]

    if b1_mag.min() < 0:
        raise ValueError("Unexpected negative magnitude values")

    # Phase values are stored in the second half of the 4th dimension. Siemens phase range: [248 - 3848]
    b1_phase = image[:, :, :, n_channels:]

    # Check that the 'ImageOrientationPatientDICOM' tag exists in the JSON file
    if 'ImageOrientationPatientDICOM' not in json_data:
        raise KeyError("Missing JSON tag: 'ImageOrientationPatientDICOM'. Check dcm2niix version.")

    # Reorder data shuffled by dm2niix into shape (x, y , n_slices*n_channels)
    b1_mag_vector = np.zeros((image.shape[0], image.shape[1], n_slices * n_channels))
    b1_phase_vector = np.zeros((image.shape[0], image.shape[1], n_slices * n_channels))
    if 'ImageOrientationText' in json_data:
        orientation = json_data['ImageOrientationText'].upper()
    else:
        logger.info("No 'ImageOrientationText' tag. Slice orientation determined from 'ImageOrientationPatientDICOM'.")
        orientation = get_main_orientation(json_data['ImageOrientationPatientDICOM'])

    # Axial or coronal cases (+ tilted)
    if orientation[:3] in ['TRA', 'COR']:
        for i in range(n_channels):
            b1_mag_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_mag[:, :, :, i]
            b1_phase_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_phase[:, :, :, i]
    # Sagittal case (+ tilted)
    elif orientation[:3] == 'SAG':
        for i in range(n_channels):
            b1_mag_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_mag[:, :, ::-1, i]
            b1_phase_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_phase[:, :, ::-1, i]
    else:
        raise ValueError("Unknown slice orientation")

    # Reorder data shuffled by dm2niix into shape (x, y, n_slices, n_channels)
    b1_mag_ordered = np.zeros_like(b1_mag)
    b1_phase_ordered = np.zeros_like(b1_phase)

    for i in range(n_slices):
        b1_mag_ordered[:, :, i, :] = b1_mag_vector[:, :, i * n_channels:i * n_channels + n_channels]
        b1_phase_ordered[:, :, i, :] = b1_phase_vector[:, :, i * n_channels:i * n_channels + n_channels]

    # Scale magnitude in nT/V
    b1_mag_ordered = b1_mag_ordered / 10  # Siemens magnitude values are stored in degrees x10
    b1_mag_ordered[b1_mag_ordered > 180] = 180  # Values higher than 180 degrees are due to noise
    # Calculate B1+ efficiency (1ms, pi-pulse) and scale by the ratio of the measured FA to the saturation FA.
    # Get the Transmission amplifier reference amplitude
    amplifier_voltage = json_data['TxRefAmp']  # [V]
    socket_voltage = amplifier_voltage * 10 ** -0.095  # -1.9dB voltage loss from amplifier to coil socket
    b1_mag_ordered = (b1_mag_ordered / SATURATION_FA) * (np.pi / (GAMMA * socket_voltage * 1e-3)) * 1e9  # nT/V

    # Scale the phase between [-pi, pi]
    # Remove potential out of range zeros (set them as null phase = 2048)
    b1_phase_ordered[b1_phase_ordered == 0] = 2048
    b1_phase_ordered = (b1_phase_ordered - 2048) * np.pi / 1800  # [-pi pi]

    # Compute the corrected complex B1+ maps
    b1_complex = b1_mag_ordered * np.exp(1j * b1_phase_ordered)

    # Modify header tags
    nii_b1.header['datatype'] = 32  # 32 corresponds to complex data
    nii_b1.header['aux_file'] = 'Uncombined B1+ maps'

    # tfl_rfmap yields bogus affine matrices that need to be fixed to visualize the B1+ maps in FSLeyes
    qfac = nii_b1.header['pixdim'][0]

    # These values are inverted in ImageOrientationPatientDICOM. Correcting them fixes the affine matrix
    json_data['ImageOrientationPatientDICOM'][0] = -json_data['ImageOrientationPatientDICOM'][0]
    json_data['ImageOrientationPatientDICOM'][1] = -json_data['ImageOrientationPatientDICOM'][1]
    json_data['ImageOrientationPatientDICOM'][5] = -json_data['ImageOrientationPatientDICOM'][5]

    xa, xb, xc, ya, yb, yc = np.asarray(json_data['ImageOrientationPatientDICOM'])

    # Compute the rotation matrix from the corrected values
    R = [[xa, ya, qfac * (xb * yc - xc * yb)],
         [xb, yb, qfac * (xc * ya - xa * yc)],
         [xc, yc, qfac * (xa * yb - xb * ya)]]

    # Build the corrected affine matrix
    affine = np.zeros((4, 4))
    affine[:3, :3] = R * nii_b1.header['pixdim'][1:4]
    affine[3, :] = [0, 0, 0, 1]
    affine[:3, 3] = [nii_b1.header['qoffset_x'], nii_b1.header['qoffset_y'], nii_b1.header['qoffset_z']]

    # Return a fixed NIfTI object with the corrected affine matrix and the reshuffled/rescaled B1+ maps
    return nib.Nifti1Image(b1_complex, affine, header=nii_b1.header)
