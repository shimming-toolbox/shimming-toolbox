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
SECONDS_IN_A_DAY = 24 * 60 * 60
ERROR_MARGIN = 0.99


def get_acquisition_times(nii_data, json_data, when='slice-middle'):
    f""" Return the acquisition timestamps from a json sidecar. This assumes BIDS convention.

    Args:
        nii_data (nibabel.Nifti1Image): Nibabel object containing the image timeseries.
        json_data (dict): Json dict corresponding to a nifti sidecar.
        when (str): When to get the acquisition time. Can be within {POSSIBLE_TIMINGS}.

    Returns:
        numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
    """

    if when not in POSSIBLE_TIMINGS:
        raise ValueError(f"Invalid 'when' parameter. Must be within {POSSIBLE_TIMINGS}")

    # Get number of volumes
    n_volumes = nii_data.header['dim'][4]
    n_slices = nii_data.shape[2]

    deltat_volume = get_volume_tr(json_data)

    # Time the acquisition of data that this image started (ISO format)
    acq_start_time_ms = get_acquisition_start_time(json_data)

    # Start time for each volume [ms]
    volume_start_times = np.linspace(acq_start_time_ms,
                                     ((n_volumes - 1) * deltat_volume) + acq_start_time_ms,
                                     n_volumes)

    if when == 'volume-start':
        return volume_start_times
    if when == 'volume-middle':
        return volume_start_times + (deltat_volume / 2)

    try:
        slice_timing_middle = get_phase_encode_0_crossings(json_data, n_slices)
    except Exception as e:
        logger.error(f"Error while getting slice timing: {e}")
        raise e
        slice_timing_middle = np.zeros(n_slices)

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


def get_slice_tr(json_data):
    """ Get the slice repetition time in milliseconds from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Slice repetition time in milliseconds.
    """
    excitation_tr = get_excitation_tr(json_data)

    pulse_sequence_details = json_data.get('PulseSequenceDetails')
    if pulse_sequence_details is None:
        raise ValueError("No PulseSequenceDetails name found in JSON file.")

    n_phase_encode_steps = get_n_acquired_phase_encode_lines(json_data)

    if "gre_field_mapping" in pulse_sequence_details:
        # This protocol acquires 2 echos with 2 different TRs
        deltat_slice = excitation_tr * n_phase_encode_steps * 2
        logging.info("Detected field mapping sequence, doubling the slice duration (1 RF pulse per echo).")
    elif ("%SiemensSeq%\\gre" == pulse_sequence_details or
          "gre_PMUlog" in pulse_sequence_details or
          pulse_sequence_details == '%CustomerSeq%\\gre_shimming'):
        deltat_slice = excitation_tr * n_phase_encode_steps
    else:
        raise NotImplementedError("Protocol name not recognized.")

    return deltat_slice


def get_phase_encode_0_crossings(data, n_sli):
    """ Return the best guess of when the middle of k-space was acquired for each slice.
    This is implemented for Siemens, 2D acquisitions and tags found in the JSON sidecar

    Args:
        data (dict): Json dict corresponding to a nifti sidecar.
        n_sli (int): Number of slices in the volume.

    Returns:
        np.ndarray: Slice timing in ms (n_slices).
    """

    # Can be '2D'
    mr_acquisition_type = data.get('MRAcquisitionType')
    if mr_acquisition_type != '2D':
        # mr_acquisition_type is None or 3D
        raise NotImplementedError("MR acquisition type is not 2D.")

    # list containing the time at which each slice was acquired
    slice_timing_start = get_slice_timing(data, n_sli)

    deltat_slice = get_slice_tr(data)

    # Error check
    volume_tr = get_volume_tr(data)

    if (deltat_slice * n_sli) * ERROR_MARGIN > volume_tr:
        ValueError("Slice timing of slices is longer than the volume timing.")

    # Get when the middle of k-space was acquired
    fourier = get_partial_fourier(data)

    # The crossing of the center-line of k-space in the phase encoding direction is not exactly at 1/2
    # of the time it takes to acquire a slice if partial Fourier is not 1. For a 7/8 partial Fourier,
    # it would acquire the center-line of k-space after 3/7 of the time it takes to acquire a slice.
    # (For a 700ms slice, the center-line would be acquired ~300ms). I simulated a couple of sequences in
    # POET and observed that the "smaller" portion is always acquired first (the 1/8 portion is skipped,
    # then 3/8 is acquired, k-space is crossed then the final 1/2 is acquired. In total, the k-space crossing
    # was at 3/7 of the slice time).

    ratio = (fourier - 0.5) / fourier

    slice_timing_mid = slice_timing_start + (deltat_slice * ratio)

    return slice_timing_mid


def get_slice_timing(json_data, n_slices):
    """ Get the slice timing from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.
        n_slices (int): Number of slices in the volume.

    Returns:
        numpy.ndarray: Slice timing in ms (n_slices).
    """
    # list containing the time at which each slice was acquired
    slice_timing_start = json_data.get('SliceTiming')
    if slice_timing_start is None:
        if n_slices == 1:
            # Slice timing information does not seem to be defined if there is only one slice
            slice_timing_start = [0.0]
        else:
            raise ValueError("No slice timing information found in JSON file.")
    else:
        slice_timing_start = np.array(slice_timing_start) * 1000  # [ms]

    return np.array(slice_timing_start)  # [ms]


def get_n_acquired_phase_encode_lines(json_data):
    """ Get the number of acquired phase encoding lines from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        int: Number of acquired phase encoding lines.
    """

    # On a Siemens scanner, I simulated many acquistions to figure out how many phase encoding lines get acquired
    # depending on sequence parameters. Here is a summary of the results:
    #   Basematrix  |   IPAT  | Ref lines | Partial Fourier | Phase oversampling | -> Acquired PE lines
    #   64          | 2       | 24        | 1               | 0                  | 44
    #   64          | 2       | 24        | 7/8             | 0                  | 40
    #   64          | 2       | 24        | 6/8             | 0                  | 36
    #   64          | 1       | 0         | 6/8             | 0                  | 48
    #   64          | 2       | 32        | 7/8             | 0                  | 44
    #   64          | 2       | 32        | 7/8             | 0.3                | 53

    if json_data.get('Manufacturer') != 'Siemens':
        raise ValueError("This function is only implemented for Siemens scanners.")

    # Base resolution
    # PhaseEncodingSteps should include the calculations below, but I have noticed that it sometimes does not
    # phase_encode_steps = json_data.get('PhaseEncodingSteps')
    phase_encode_steps = json_data.get('AcquisitionMatrixPE')
    if phase_encode_steps is None:
        raise ValueError("AcquisitionMatrixPE not found in JSON sidecar.")
    # Convert to int if it was not None
    phase_encode_steps = int(phase_encode_steps)

    # Phase oversampling
    phase_over = json_data.get('PhaseOversampling')
    if phase_over is None:
        # If phase oversampling is not defined, assume it is 1 (no phase oversampling)
        phase_over = 1.0
    else:
        # PhaseOversampling is given as a percentage, we add 1 so it can be multiplied
        phase_over = float(phase_over) + 1.0

    # Partial Fourier
    partial_fourier = get_partial_fourier(json_data)

    n_phase_encode_steps = phase_encode_steps * partial_fourier * phase_over

    # Parallel acquisition reduction
    parallel_technique = json_data.get('ParallelAcquisitionTechnique')
    if parallel_technique is not None:

        if parallel_technique == 'GRAPPA':
            parallel_reduction_factor_in_plane = json_data.get('ParallelReductionFactorInPlane')
            if parallel_reduction_factor_in_plane is None:
                parallel_reduction_factor_in_plane = 1.0
            parallel_reduction_factor_in_plane = float(parallel_reduction_factor_in_plane)

            ref_lines_pe = json_data.get('RefLinesPE')
            if ref_lines_pe is None and parallel_reduction_factor_in_plane != 1.0:
                raise ValueError("RefLinesPE not found in JSON sidecar.")
            ref_lines_pe = int(ref_lines_pe)

            n_phase_encode_steps = math.ceil((n_phase_encode_steps + ref_lines_pe) / parallel_reduction_factor_in_plane)
        else:
            NotImplementedError(f"{parallel_technique} parallel acquisition technique is not implemented yet.")

    ph_encode_steps_dcm2niix = json_data.get('PhaseEncodingSteps')
    if ph_encode_steps_dcm2niix is not None:
        ph_encode_steps_dcm2niix = int(ph_encode_steps_dcm2niix)
        if ph_encode_steps_dcm2niix != n_phase_encode_steps:
            logger.warning(f"PhaseEncodingSteps in JSON sidecar ({ph_encode_steps_dcm2niix}) does not match "
                           f"calculated value ({n_phase_encode_steps}). This is a bug in Shimming Toolbox or in "
                           f"dcm2niix. Using {n_phase_encode_steps} phase encoding steps (from ST).")

    return n_phase_encode_steps


def get_partial_fourier(json_data):
    """ Get the partial fourier value from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Partial Fourier value between 0 and 1. If no partial fourier is defined, returns 1.0.
    """
    manufacturer = json_data.get('Manufacturer')
    if manufacturer == 'Siemens':
        partial_fourier = json_data.get('PartialFourier')
        if partial_fourier is None:
            logger.warning("No partial fourier information found in JSON file, assuming no partial fourier.")
            partial_fourier = 1.0
        elif partial_fourier < 1 / 2:
            raise ValueError("Partial fourier value is less than 1/2. That should not be possible.")
        else:
            partial_fourier = float(partial_fourier)
            if not (0 <= partial_fourier <= 1):
                raise ValueError("Partial Fourier value format not supported, make sure it is between 0 and 1.")
    else:
        NotImplementedError("Partial Fourier not implemented for this manufacturer. ")

    return partial_fourier


def get_excitation_tr(json_data):
    """ Get the slice excitation repetition time in milliseconds from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Slice excitation repetition time in milliseconds.
    """

    # If RepetitionTimeExcitation is not defined, then the RepetitionTime is the time between 2 RF pulses of the
    # same slice
    if json_data.get('RepetitionTimeExcitation') is None:
        excitation_tr = json_data.get('RepetitionTime')
        if excitation_tr is None:
            raise ValueError("RepetitionTimeExcitation nor RepetitionTime is defined in the JSON sidecar. "
                             "Can't figure out excitation TR")
    else:
        excitation_tr = json_data.get('RepetitionTimeExcitation')

    if excitation_tr is not None:
        excitation_tr = float(excitation_tr) * 1000  # [ms]

    return excitation_tr


def get_volume_tr(json_data):
    """ Get the volume repetition time in milliseconds from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Volume repetition time in milliseconds.
    """

    # If both RepetitionTimeExcitation and RepetitionTime are defined, then RepetitionTime is the volume TR and
    # RepetitionTimeExcitation is the time between 2 RF pulses of the same slice.
    # If RepetitionTimeExcitation is not defined, RepetitionTime is the time between 2 RF pulses of the same slice and
    # AcquisitionDuration is the time to acquire a single volume (i.e.: volume TR).
    if json_data.get('RepetitionTimeExcitation') is None:
        if json_data.get('AcquisitionDuration') is None:
            # deltat_slice = get_slice_tr(json_data)
            # n_slices = nii_data.shape[2]
            # deltat_volume = deltat_slice * n_slices

            raise ValueError("RepetitionTimeExcitation nor AcquisitionDuration is defined in the JSON "
                             "sidecar. Can't compute volume repetition time")
        else:
            # If AcquisitionDuration is defined and RepetitionTimeExcitation is not,
            # then AcquisitionDuration is the time to acquire a single volume
            deltat_volume = float(json_data['AcquisitionDuration']) * 1000
    else:
        # If RepetitionTimeExcitation is defined, RepetitionTime is the volume TR
        deltat_volume = float(json_data['RepetitionTime']) * 1000  # [ms]

    if deltat_volume is None:
        raise ValueError("Can't compute volume repetition time. RepetitionTimeExcitation, AcquisitionDuration and/or "
                         "RepetitionTime can't be used to determine volume TR.")

    return deltat_volume


def get_acquisition_duration(nii_data, json_data):
    """ Compute the acquisition duration from the nifti data and the json sidecar.

    Args:
        nii_data (nibabel.Nifti1Image): Nibabel object containing the image data.
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Acquisition duration in milliseconds.
    """

    if nii_data.ndim == 4:
        n_volumes = nii_data.shape[3]
    else:
        # If the data is not 4D, then there is no time dimension, so there is only a single volume
        n_volumes = 1

    deltat_volume = get_volume_tr(json_data)

    return n_volumes * deltat_volume  # [ms]


def get_acquisition_start_time(json_data):
    """
    Get the acquisition start time in milliseconds past midnight from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Acquisition start time in milliseconds past midnight.

    """
    acq_start_time_iso = json_data.get('AcquisitionTime')
    if acq_start_time_iso is None:
        raise ValueError("Acquisition time not found in json sidecar.")
    # todo: dummy scans?
    acq_start_time_ms = iso_times_to_ms(np.array([acq_start_time_iso]))[0]  # [ms]
    return acq_start_time_ms


def get_acquisition_stop_time(nii_data, json_data):
    """
    Get the acquisition stop time in milliseconds past midnight from the json sidecar.

    Args:
        json_data (dict): Json dict corresponding to a nifti sidecar.

    Returns:
        float: Acquisition stop time in milliseconds past midnight.

    """

    acq_start_time = get_acquisition_start_time(json_data)
    acq_duration = get_acquisition_duration(nii_data, json_data)
    acq_stop_time = acq_start_time + acq_duration

    # If the acquisition stop time is greater than 24 hours, then it is the next day
    return acq_stop_time % (SECONDS_IN_A_DAY * 1000)


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


def is_fatsat_on(json_data):
    """ Return if the scan was acquired using fat sat.

    Args:
        json_anat (dict): BIDS Json sidecar

    Returns:
        bool: Whether fat sat is on
    """

    if 'ScanOptions' in json_data:
        if 'FS' in json_data['ScanOptions']:
            logger.debug("Fat Saturation pulse detected")
            is_fatsat = True
        else:
            logger.debug("No Fat Saturation pulse detected")
            is_fatsat = False
    else:
        logger.warning("No ScanOptions found in JSON sidecar. Cannot determine if fat saturation is on.")
        is_fatsat = False

    return is_fatsat
