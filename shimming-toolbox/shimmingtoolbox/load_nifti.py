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
POSSIBLE_TIMINGS = ['slice-middle', 'volume-start', 'volume-middle']


def get_acquisition_times(nif_data, when='slice-middle'):
    """ Return the acquisition timestamps from a json sidecar. This assumes BIDS convention.

    Args:
        nif_data (NiftiFieldMap): NiftiFieldMap object containing the nifti data and json sidecar.
        when (str): When to get the acquisition time. Can be within {POSSIBLE_TIMINGS}.

    Returns:
        numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
    """

    if when not in POSSIBLE_TIMINGS:
        raise ValueError(f"Invalid 'when' parameter. Must be within {POSSIBLE_TIMINGS}")

    # Get number of volumes
    n_volumes = nif_data.header['dim'][4]
    n_slices = nif_data.shape[2]

    # Time between the beginning of the acquisition of a volume and the beginning of the acquisition of the next volume
    deltat_volume = float(nif_data.get_json_info('RepetitionTime')) * 1000  # [ms]

    # Time the acquisition of data for this image started (ISO format)
    acq_start_time_iso = nif_data.get_json_info('AcquisitionTime')
    # todo: dummy scans?
    acq_start_time_ms = iso_times_to_ms(np.array([acq_start_time_iso]))[0]

    # Start time for each volume [ms]
    volume_start_times = np.linspace(acq_start_time_ms,
                                     ((n_volumes - 1) * deltat_volume) + acq_start_time_ms,
                                     n_volumes)

    if when == 'volume-start':
        return volume_start_times
    if when == 'volume-middle':
        return volume_start_times + (deltat_volume / 2)

    def _get_middle_of_slice_timing(data, n_sli):
        """ Return the best guess of when the middle of k-space was acquired for each slice. Return an array of 0 if no
            best guess is found

        Args:
            data (dict): Json dict corresponding to a nifti sidecar.
            n_sli (int): Number of slices in the volume.

        Returns:
            np.ndarray: Slice timing in ms (n_slices).
        """

        # Can be '2D' or 3D
        mr_acquisition_type = data.get_json_info('MRAcquisitionType', required=False)
        if mr_acquisition_type != '2D':
            # mr_acquisition_type is None or 3D
            logger.warning("MR acquisition type is not 2D.")
            return np.zeros(n_slices)

        # list containing the time at which each slice was acquired
        slice_timing_start = data.get_json_info('SliceTiming', required=False)
        if slice_timing_start is None:
            if n_sli == 1:
                # Slice timing information does not seem to be defined if there is only one slice
                slice_timing_start = [0.0]
            else:
                logger.warning("No slice timing information found in JSON file.")
                return np.zeros(n_slices)

        # Convert to ms
        slice_timing_start = np.array(slice_timing_start) * 1000  # [ms]

        phase_encode_steps = int(data.get_json_info('PhaseEncodingSteps'))
        repetition_slice_excitation = float(data.get_json_info('RepetitionTimeExcitation'))
        if repetition_slice_excitation is None or phase_encode_steps is None:
            logger.warning("Not enough information to figure out each slice time. "
                           "Either/Both of RepetitionTimeExcitation and PhaseEncodingSteps is/are undefined")
            return np.zeros(n_slices)

        # If the slice timing is lower than the TR excitation, this is an interleaved multi-slice acquisition
        # (more than one slice acquired within a TR excitation)
        repetition_slice_excitation = repetition_slice_excitation * 1000  # [ms]
        # Remove slices that are at 0 ms (acquired during the first TR excitation)
        slice_timing_start_no_zero = slice_timing_start[slice_timing_start > 0]
        # If there are other slices acquired before the next TR excitation, then this is interleaved
        # multi-slice
        if np.any(slice_timing_start_no_zero < repetition_slice_excitation):
            logger.warning("Interleaved multi-slice acquisition detected.")
            return np.zeros(n_slices)

        pulse_sequence_details = data.get_json_info('PulseSequenceDetails')
        if pulse_sequence_details is None:
            logger.warning("No PulseSequenceDetails name found in JSON file.")
            return np.zeros(n_slices)

        if "%CustomerSeq%\\gre_field_mapping_PMUlog" == pulse_sequence_details:
            # This protocol acquires 2 echos with 2 different TRs
            deltat_slice = repetition_slice_excitation * phase_encode_steps * 2
        elif "%SiemensSeq%\\gre" == pulse_sequence_details or "gre_PMUlog" in pulse_sequence_details:
            deltat_slice = repetition_slice_excitation * phase_encode_steps
        else:
            logger.warning("Protocol name not recognized.")
            return np.zeros(n_slices)

        # Error check
        volume_tr = data.get_json_info('RepetitionTime')
        if volume_tr is None:
            logger.warning("No volume 'RepetitionTime' found in JSON file.")
            return np.zeros(n_slices)
        deltat_vol = float(volume_tr) * 1000  # [ms]
        if (deltat_slice * n_sli) > deltat_vol:
            logger.warning("Slice timing of slices is longer than the volume timing.")
            return np.zeros(n_slices)

        # Get when the middle of k-space was acquired
        manufacturer = data.get_json_info('Manufacturer', required=False)
        if manufacturer == 'Siemens':
            fourier = data.get_json_info('PartialFourier')
            if fourier is None:
                logger.warning("No partial fourier information found in JSON file, assuming no partial fourier.")
                fourier = 1.0
            elif fourier < 1/2:
                raise ValueError("Partial fourier value is less than 1/2. That should not be possible.")
            else:
                fourier = float(fourier)
                if not (0 <= fourier <= 1):
                    logger.warning("Partial Fourier value format not supported, make sure it is between 0 and 1.")
                    return np.zeros(n_slices)

            # The crossing of the center-line of k-space in the phase encoding direction is not exactly at 1/2
            # of the time it takes to acquire a slice if partial Fourier is not 1. For a 7/8 partial Fourier,
            # it would acquire the center-line of k-space after 3/7 of the time it takes to acquire a slice.
            # (For a 700ms slice, the center-line would be acquired ~300ms). I simulated a couple of sequences in
            # POET and observed that the "smaller" portion is always acquired first (the 1/8 portion is skipped,
            # then 3/8 is acquired, k-space is crossed then the final 1/2 is acquired. In total, the k-space crossing
            # was at 3/7 of the slice time).

            ratio = (fourier - 0.5) / fourier
        else:
            logger.warning("Manufacturer not supported for partial fourier, assuming no partial fourier.")
            ratio = 0.5

        slice_timing_mid = slice_timing_start + (deltat_slice * ratio)

        return slice_timing_mid

    slice_timing_middle = _get_middle_of_slice_timing(nif_data, n_slices)
    timing = np.zeros((n_volumes, n_slices))
    if np.all(slice_timing_middle == 0):
        logger.warning("Could not figure out the slice timing. Using one time-point per volume instead.")
        # If the slice timing is set to 0, then we could not figure out the slice timing, then set the best guess to
        # the time required to get to the middle of the volume
        for i in range(n_volumes):
            timing[i, :] = np.repeat(volume_start_times[i] + (deltat_volume / 2), n_slices)
    else:
        # If we figured out the slice timing, then we can use it to get the timing of each slice for each volume
        for i in range(n_volumes):
            timing[i, :] = volume_start_times[i] + slice_timing_middle  # [ms]

    return timing  # [ms]


def load_nifti(path_data, modality='phase'):
    """ Load data from a directory containing NIFTI type file with nibabel.

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
