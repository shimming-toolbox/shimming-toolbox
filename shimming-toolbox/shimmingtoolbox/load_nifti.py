#!usr/bin/env python3
# -*- coding: utf-8

import json
import logging
import math
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.utils import iso_times_to_ms

logger = logging.getLogger(__name__)
PHASE_SCALING_SIEMENS = 4096


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
        and 2pi.
    """
    nii = nib.load(fname_nifti)
    json_path = fname_nifti.split('.nii')[0] + '.json'

    if os.path.isfile(json_path):
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        raise OSError("Missing json file")
    # Store nifti image in a numpy array
    image = np.asarray(nii.dataobj)
    if auto_scale:
        logger.info("Scaling the selected NIfTI")

        # If B0 phase maps
        if json_data.get('Manufacturer') == 'Siemens' and \
                (('ImageComments' in json_data) and ("*phase*" in json_data['ImageComments'])
                 or ('ImageType' in json_data) and ('P' in json_data['ImageType'])):
            # Rescales from -pi to pi
            extent = (np.amax(image) - np.amin(image))

            if np.amin(image) < 0 and (0.95 * 2 * PHASE_SCALING_SIEMENS < extent < 2 * PHASE_SCALING_SIEMENS * 1.05):
                # Siemens' scaling: [-4096, 4095] --> [-pi, pi[
                image = (image * math.pi / PHASE_SCALING_SIEMENS)
            elif np.amin(image) >= 0 and (0.95 * PHASE_SCALING_SIEMENS < extent < PHASE_SCALING_SIEMENS * 1.05):
                # Siemens' scaling [0, 4095] --> [-pi, pi[
                image = (image * 2 * math.pi / PHASE_SCALING_SIEMENS) - math.pi
            else:
                logger.info("Could not scale phase data")

            # Create new nibabel object with updated image
            nii = nib.Nifti1Image(image, nii.affine, header=nii.header)
        elif json_data.get('Manufacturer') == 'Philips' \
                and (('ImageType' in json_data) and (('Phase' in json_data['ImageType']) or
                                                     ('P' in json_data['ImageType']))):

            extent = (np.amax(image) - np.amin(image))
            if np.amin(image) < 0 and np.amax(image) and (0.95 * math.pi < extent < 1.05 * 2 * math.pi):
                # Philips scaling: [-pi, pi)
                pass
            else:
                logger.info("Could not scale phase data")
        elif json_data.get('Manufacturer') == 'GE':
            logger.info("GE phase scaling information is not yet implemented")
        else:
            logger.info("Unknown NIfTI type: No scaling applied")

    else:
        pass

    return nii, json_data, image


def get_isocenter(json_data):
    """ Get the isocenter location in RAS coordinates from the json file. The patient position is used to infer the
    table position in the patient coordinate system. When the table is at (0,0,0), the origin is at the isocenter.
    We can therefore infer the isocenter as -table_position when the table_position is in RAS coordinates.

    Args:
        json_data (dict): Dictionary containing the BIDS sidecar information

    Returns:
        numpy.ndarray: Isocenter location in RAS coordinates
    """
    table_position = json_data.get('TablePosition')
    if table_position is None:
        raise ValueError("Table position not found in json sidecar.")

    patient_position = json_data.get('PatientPosition')
    if patient_position is None:
        raise ValueError("Patient position not found in json sidecar.")

    # From the Bid specification "TablePosition":
    # The table position, relative to an implementation-specific reference point, often the isocenter. Values must be
    # an array (1x3) of three distances in millimeters in absolute coordinates (world coordinates). If an observer
    # stands in front of the scanner looking at it, a table moving to the left, up or into the scanner (from the
    # observer's point of view) will increase the 1st, 2nd and 3rd value in the array respectively. The origin is
    # defined by the image affine.
    table_position = np.array(table_position)

    # Convert table position to RAS coordinates
    table_position_ras = np.zeros(3)
    if patient_position == 'HFS':
        table_position_ras = table_position
    elif patient_position == 'HFP':
        table_position_ras[0] = -table_position[0]
        table_position_ras[1] = -table_position[1]
        table_position_ras[2] = table_position[2]
    elif patient_position == 'FFS':
        table_position_ras[0] = -table_position[0]
        table_position_ras[1] = table_position[1]
        table_position_ras[2] = -table_position[2]
    elif patient_position == 'FFP':
        table_position_ras[0] = table_position[0]
        table_position_ras[1] = -table_position[1]
        table_position_ras[2] = -table_position[2]
    elif patient_position == 'LFP':
        table_position_ras[0] = -table_position[2]
        table_position_ras[1] = -table_position[1]
        table_position_ras[2] = -table_position[0]
    elif patient_position == 'LFS':
        table_position_ras[0] = -table_position[2]
        table_position_ras[1] = table_position[1]
        table_position_ras[2] = table_position[0]
    elif patient_position == 'RFP':
        table_position_ras[0] = table_position[2]
        table_position_ras[1] = -table_position[1]
        table_position_ras[2] = table_position[0]
    elif patient_position == 'RFS':
        table_position_ras[0] = table_position[2]
        table_position_ras[1] = table_position[1]
        table_position_ras[2] = -table_position[0]
    elif patient_position == 'HFDR':
        table_position_ras[0] = -table_position[1]
        table_position_ras[1] = table_position[0]
        table_position_ras[2] = table_position[2]
    elif patient_position == 'HFDL':
        table_position_ras[0] = table_position[1]
        table_position_ras[1] = -table_position[0]
        table_position_ras[2] = table_position[2]
    elif patient_position == 'FFDR':
        table_position_ras[0] = -table_position[1]
        table_position_ras[1] = -table_position[0]
        table_position_ras[2] = -table_position[2]
    elif patient_position == 'FFDL':
        table_position_ras[0] = table_position[1]
        table_position_ras[1] = table_position[0]
        table_position_ras[2] = -table_position[2]
    elif patient_position == 'AFDR':
        table_position_ras[0] = -table_position[1]
        table_position_ras[1] = table_position[2]
        table_position_ras[2] = -table_position[0]
    elif patient_position == 'AFDL':
        table_position_ras[0] = table_position[1]
        table_position_ras[1] = table_position[2]
        table_position_ras[2] = table_position[0]
    elif patient_position == 'PFDR':
        table_position_ras[0] = -table_position[1]
        table_position_ras[1] = -table_position[2]
        table_position_ras[2] = table_position[0]
    elif patient_position == 'PFDL':
        table_position_ras[0] = table_position[1]
        table_position_ras[1] = -table_position[2]
        table_position_ras[2] = -table_position[0]
    else:
        raise NotImplementedError(f"Patient position {patient_position} not implemented")

    # The isocenter is located at -table_position
    isocenter = -table_position_ras

    return isocenter
