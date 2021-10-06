#!usr/bin/env python3
# -*- coding: utf-8

import os
import logging
import numpy as np
import nibabel as nib
import json
import math
import warnings

from shimmingtoolbox.utils import iso_times_to_ms

logger = logging.getLogger(__name__)
PHASE_SCALING_SIEMENS = 4096
GAMMA = 2.675e8  # Proton's gyromagnetic ratio (rad/(T.s))
SATURATION_FA = 90  # Saturation flip angle hard-coded in TFL B1 mapping sequence (deg)


def get_acquisition_times(nii_data, json_data):
    """Return the acquisition timestamps from a json sidecar. This assumes BIDS convention.

    Args:
        nii_data (nibabel.Nifti1Image): Nibabel object containing the image timeseries.
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        numpy.ndarray: Acquisition timestamps in ms.

    """
    # Get number of volumes
    n_volumes = nii_data.header['dim'][4]

    delta_t = json_data['RepetitionTime'] * 1000  # [ms]
    acq_start_time_iso = json_data['AcquisitionTime']  # ISO format
    acq_start_time_ms = iso_times_to_ms(np.array([acq_start_time_iso]))[0]  # [ms]

    return np.linspace(acq_start_time_ms, ((n_volumes - 1) * delta_t) + acq_start_time_ms, n_volumes)  # [ms]


def load_nifti(path_data, modality='phase'):
    """Load data from a directory containing NIFTI type file with nibabel.
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

    # Generate file_list
    file_list = []
    [file_list.append(os.path.join(path_data, f)) for f in os.listdir(path_data) if f not in file_list]
    file_list = sorted(file_list)

    nifti_path = ""
    # Check for incompatible acquisition source path
    if all([os.path.isdir(f) for f in file_list]):
        acquisitions = [f for f in file_list if os.path.isdir(f)]
        logger.info("Multiple acquisition directories in path. Choosing only one.")
    elif all([os.path.isfile(f) for f in file_list]):
        logger.info("Acquisition directory given. Using acquisitions.")
        nifti_path = path_data
    else:
        raise RuntimeError("Directories and files in input path")

    # Choose an acquisition between all folders
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
                logger.error(f"Input must be linked to an acquisition folder. {input_resp} is out of range")

        nifti_path = os.path.abspath(file_list[select_acquisition])

    # Get a list of nii files
    nifti_list = [os.path.join(nifti_path, f) for f in os.listdir(nifti_path) if f.endswith((".nii", ".nii.gz"))]
    nifti_list = sorted(nifti_list)

    # Read all images and headers available and store them
    nifti_init = [read_nii(nifti_list[i]) for i in range(len(nifti_list))]

    info = []
    json_info = []
    niftis = np.empty([1, 1], dtype=float)

    run_list = {}
    # Parse and separate each file by run sequence with modality check
    for file_info in nifti_init:
        if file_info[1]['AcquisitionNumber'] not in run_list.keys():
            run_list[file_info[1]['AcquisitionNumber']] = []
        run_list[file_info[1]['AcquisitionNumber']].append((file_info, file_info[1]['ImageComments']))

    # If more than one run, select one
    select_run = -1
    if len(list(run_list.keys())) > 1:
        for i in list(run_list.keys()):
            print(f"{i}\n")

        while 1:
            input_resp = input("Enter the number for the appropriate run number, (type 'q' to quit) : ")
            if input_resp == 'q':
                return 0
            select_run = int(input_resp)

            if select_run in list(run_list.keys()):
                break
            else:
                logger.error(f"Input must be linked to a run number. {input_resp} is out of range")
    else:
        select_run = list(run_list.keys())[0]
        logger.info(f"Reading acquisitions for run {list(run_list.keys())[0]}")

    # Create output array and headers
    nifti_pos = 0
    echo_shape = sum(1 for tmp_info in run_list[select_run] if modality in tmp_info[1])

    niftis = np.empty([nifti_init[0][0].shape[0], nifti_init[0][0].shape[1], nifti_init[0][0].shape[2], echo_shape,
                       (1 if nifti_init[0][0].ndim == 3 else nifti_init[0][0].shape[3])], dtype=float)

    for i_echo in range(len(run_list[select_run])):
        tmp_nii = run_list[select_run][i_echo][0]
        if modality in run_list[select_run][i_echo][1]:
            info.append(tmp_nii[0].header)
            json_info.append(tmp_nii[1])
            if niftis.shape[4] == 1:
                niftis[:, :, :, nifti_pos, 0] = tmp_nii[2]
            else:
                for i_volume in range(nifti_init[0][0].shape[3]):
                    niftis[:, :, :, nifti_pos, i_volume] = tmp_nii[2][:, :, :, i_volume]
            nifti_pos += 1

    return niftis, info, json_info


def read_nii(fname_nifti, auto_scale=True):
    """Reads a nifti file and returns the corresponding image and info. Also returns the associated json data.
    Args:
        fname_nifti (str): direct path to the .nii or .nii.gz file that is going to be read
        auto_scale (:obj:`bool`, optional): Tells if scaling is done before return
    Returns:
        info (Nifti1Image): Objet containing various data about the nifti file (returned by nibabel.load)
        json_data (dict): Contains the different fields present in the json file corresponding to the nifti file
        image (numpy.ndarray): For B0-maps, image contained in the nifti. Siemens phase images are rescaled between 0
        and 2pi. For RF-maps, complex array of dimension (x, y, slice, coil) with phase between -pi and pi.
    """

    info = nib.load(fname_nifti)
    # `extractBefore` should get the correct filename in both.nii and.nii.gz cases
    json_path = fname_nifti.split('.nii')[0] + '.json'

    if os.path.isfile(json_path):
        json_data = json.load(open(json_path))
    else:
        raise OSError("Missing json file")

    # Store nifti image in a numpy array
    image = np.asarray(info.dataobj)

    if auto_scale:
        logger.info("Scaling the selected nifti")
        # If Siemens' TurboFLASH B1 mapping (dcm2niix cannot separate phase and magnitude for this sequence)
        if ('SequenceName' in json_data) and 'tfl2d1_16' in json_data['SequenceName']:
            image = scale_tfl_b1(image, json_data)

        # If B0 phase maps
        elif ('Manufacturer' in json_data) and (json_data['Manufacturer'] == 'Siemens') \
                and (('ImageComments' in json_data) and ("*phase*" in json_data['ImageComments'])
                     or ('ImageType' in json_data) and ('P' in json_data['ImageType'])):
            # Bootstrap, rescales from -pi to pi
            if np.amin(image) < 0:
                image = image * (2 * math.pi / (PHASE_SCALING_SIEMENS * 2))
            else:
                image = image * (2 * math.pi / PHASE_SCALING_SIEMENS) - math.pi
        else:
            logger.info("Unknown nifti type: No scaling applied")
    else:
        logger.info("No scaling applied to selected nifti")

    return info, json_data, image


def scale_tfl_b1(image, json_data):
    """Rescales the magnitude and phase of complex B1 maps acquired with Siemens' standard B1 mapping sequence. Also
    reorders the data that are shuffled during DICOM to NIfTI conversion.
    Args:
        image: Array of dimension (x, y, n_slices, 2*n_channels)
                    First half: magnitude in degree*10
                    Second half: phase in range [248-3848]
        json_data: Contains the different fields present in the json file corresponding to the nifti file.

    Returns:
        numpy.ndarray: complex rescale B1 maps (x, y, n_slices, n_channels)

    """
    if 'ShimSetting' in json_data:
        n_channels = len(json_data['ShimSetting'])
        if image.shape[3] != 2 * n_channels:
            raise ValueError("Wrong array dimension: number of channels not matching")
    else:
        raise KeyError("Missing json tag: 'ShimSetting'")

    if 'SliceTiming' in json_data:
        n_slices = len(json_data['SliceTiming'])
        if image.shape[2] != n_slices:
            raise ValueError("Wrong array dimension: number of slices not matching")
    else:
        warnings.warn("Missing json tag: 'SliceTiming', slices number cannot be checked.")
        n_slices = image.shape[2]

    # Magnitude values are stored in the first half of the 4th dimension
    b1_mag = image[:, :, :, :image.shape[3] // 2]

    if b1_mag.min() < 0:
        raise ValueError("Unexpected negative magnitude values")

    # Phase values are stored in the second half of the 4th dimension. Siemens phase range: [248 - 3848]
    b1_phase = image[:, :, :, image.shape[3] // 2:]

    # Reorder data shuffled by dm2niix into shape (x, y , n_slices*n_channels)
    b1_mag_vector = np.zeros((image.shape[0], image.shape[1], n_slices * n_channels))
    b1_phase_vector = np.zeros((image.shape[0], image.shape[1], n_slices * n_channels))
    if 'ImageOrientationPatientDICOM' in json_data:
        orientation = json_data['ImageOrientationPatientDICOM']
        # Axial or Coronal cases
        if orientation[:3] == [1, 0, 0]:
            for i in range(n_channels):
                b1_mag_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_mag[:, :, :, i]
                b1_phase_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_phase[:, :, :, i]
        # Sagittal case
        elif orientation[:3] == [0, 1, 0] and orientation[5] != 0:
            for i in range(n_channels):
                b1_mag_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_mag[:, :, ::-1, i]
                b1_phase_vector[:, :, i * n_slices:(i + 1) * n_slices] = b1_phase[:, :, ::-1, i]
        else:
            raise ValueError("Unknown slice orientation")
    else:
        raise KeyError("Missing json tag: 'ImageOrientationPatientDICOM'")

    # Reorder data shuffled by dm2niix into shape (x, y, n_slices, n_channels)
    b1_mag_ordered = np.zeros_like(b1_mag)
    b1_phase_ordered = np.zeros_like(b1_phase)

    for i in range(n_slices):
        b1_mag_ordered[:, :, i, :] = b1_mag_vector[:, :, i * n_channels:i * n_channels + n_channels]
        b1_phase_ordered[:, :, i, :] = b1_phase_vector[:, :, i * n_channels:i * n_channels + n_channels]

    # Scale magnitude in nT/V
    b1_mag_ordered = b1_mag_ordered / 10  # Siemens magnitude values are stored in degrees x10
    b1_mag_ordered[b1_mag_ordered > 180] = 180  # Values higher than 180 degrees are due to noise
    # Calculate B1 efficiency (1ms, pi-pulse) and scale by the ratio of the measured FA to the saturation FA.
    # Get the Transmission amplifier reference amplitude
    amplifier_voltage = json_data['TxRefAmp']  # [V]
    socket_voltage = amplifier_voltage * 10 ** -0.095  # -1.9dB voltage loss from amplifier to coil socket
    b1_mag_ordered = (b1_mag_ordered / SATURATION_FA) * (np.pi / (GAMMA * socket_voltage * 1e-3)) * 1e9  # nT/V

    # Scale the phase between [-pi, pi]
    # Remove potential out of range zeros (set them as null phase = 2048)
    b1_phase_ordered[b1_phase_ordered == 0] = 2048
    b1_phase_ordered = (b1_phase_ordered - 2048) * np.pi / 1800  # [-pi pi]

    # Return complex B1 maps
    return b1_mag_ordered * np.exp(1j * b1_phase_ordered)
