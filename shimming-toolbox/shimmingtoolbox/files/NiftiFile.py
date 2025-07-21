#!/usr/bin/python3
# -*- coding: utf-8 -*

from __future__ import annotations
from functools import wraps
import json
import logging
import math
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.utils import iso_times_to_ms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')
DEFAULT_SUFFIX = '_saved.nii.gz'

POSSIBLE_TIMINGS = ['slice-middle', 'volume-start', 'volume-middle']
SECONDS_IN_A_DAY = 24 * 60 * 60
ERROR_MARGIN = 0.99


def safe_getter(default_value=None):
    """Decorator that catches errors in getter functions and returns a default value."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.debug(f"{func.__name__}: {e}")
                # terminate the program if the error is critical
                if isinstance(e, (KeyError, NameError, ValueError, OSError)):
                    raise e
                return default_value

        return wrapper

    return decorator


class NiftiFile:
    def __init__(self, fname_nii: str, json: dict = None, path_output: str = None, json_needed: bool = True) -> None:
        if not isinstance(fname_nii, str):
            raise TypeError("fname_nii must be a string")
        if not any(fname_nii.endswith(ext) for ext in NIFTI_EXTENSIONS):
            raise ValueError(f"File must end with one of {NIFTI_EXTENSIONS}")

        # Convert relative path to absolute path
        self.fname_nii: str = os.path.abspath(fname_nii)
        self.path_nii: str = self.get_path_nii()
        self.nii: nib.Nifti1Image
        self.data: np.ndarray
        self.nii, self.data = self.load_nii()
        self.filename: str = self.get_filename()
        self.json: dict = json if json is not None else self.load_json(json_needed)
        self.header = self.nii.header
        self.affine = self.nii.affine
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.path_output = path_output if path_output else self.path_nii

    def __eq__(self, other: nib.Nifti1Image) -> NiftiFile:
        """Override the = operator to set NiftiFile data from a nibabel image.

        Args:
            other (nib.Nifti1Image): The NiBabel image to set

        Raises:
            TypeError: If other is not a nibabel.Nifti1Image

        Returns:
            NiftiFile: Self for method chaining
        """
        self.set_nii(other)
        return self  # Return self for method chaining

    def load_nii(self) -> list:
        """ Load a NIfTI file and return the NIfTI object and its data.

        Raises:
            ValueError: If the provided path does not exist or is not a valid NIfTI file.

        Returns:
            nib.Nifti1Image: The loaded NIfTI image object.
            numpy.ndarray: The data contained in the NIfTI file.
        """
        if not os.path.exists(self.fname_nii):
            raise ValueError("Not an existing NIFTI path")
        nii = nib.load(self.fname_nii)
        data = np.asanyarray(nii.dataobj)

        return nii, data

    def load_json(self, json_needed: bool = True) -> dict | None:
        """ Load the JSON file corresponding to the NIfTI file.
        The JSON file is expected to be in the same directory as the NIfTI file
        and have the same base name.

        Args:
            json_needed (bool): Specifies whether the JSON file is required.

        Returns:
            dict: The content of the JSON file if found, otherwise None.
        """
        json_path = self.get_json(json_needed)
        if json_path is not None:
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            return None

    def save(self, fname: str = None) -> None:
        """ Save the NIfTI file to a specified path.
        If no output path is provided, it saves the file in the same directory with a default name.

        Args:
            fname (str, optional): The path where the NIfTI file should be saved.
                                   If None, it saves the file in the same directory with a default name.

        Raises:
            ValueError: If the output path is not a valid directory.

        Returns:
            None: The function saves the NIfTI file to the specified path.
        """
        if fname is not None:
            if fname[-4:] != '.nii' and fname[-7:] != '.nii.gz':
                if len(fname.split('.')) > 1:
                    raise ValueError("File name must end with .nii or .nii.gz")
                else:
                    fname += ".nii.gz"
            fname_output = os.path.join(self.path_output, fname)
        else:
            fname_output = os.path.join(self.path_output, f"{self.filename}{DEFAULT_SUFFIX}")

        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        elif not os.path.isdir(self.path_output):
            raise ValueError(f"Output path {fname_output} is not a valid directory.")

        if self.json is not None:
            # Save json
            fname_json = fname_output.rsplit('.nii', 1)[0] + '.json'
            with open(fname_json, 'w') as outfile:
                json.dump(self.json, outfile, indent=2)

        logger.info(f"Saving NIfTI file to {fname_output}")
        nib.save(self.nii, fname_output)

    def set_nii(self, nii: nib.Nifti1Image) -> None:
        """ Set the NIfTI image object and its data.

        Args:
            nii (nib.Nifti1Image): The NIfTI image object to set.

        Raises:
            TypeError: If the provided nii is not a nib.Nifti1Image object.
        """
        if not isinstance(nii, nib.Nifti1Image):
            raise TypeError("nii must be a nib.Nifti1Image object")
        self.nii = nii
        self.data = np.asanyarray(nii.dataobj)
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    @safe_getter(default_value=None)
    def get_json(self, json_needed: bool = True) -> str | None:
        """ Find the corresponding JSON file for the NIfTI file.
        The JSON file is expected to be in the same directory as the NIfTI file
        and have the same base name.

        Args:
            json_needed (bool): Specifies whether the JSON file is required.

        Returns:
            str: The path to the JSON file if found, otherwise None.
        """
        fname_json = os.path.join(self.path_nii, self.filename + ".json")
        if os.path.exists(fname_json):
            return fname_json
        elif json_needed:
            raise OSError(f"JSON file not found for {self.fname_nii}. Expected at {fname_json}. ")
        else:
            return None

    @safe_getter(default_value=None)
    def get_filename(self) -> str:
        """ Get the filename without the extension from the NIfTI file path.
        Verifies that the file has a valid NIfTI extension (.nii or .nii.gz).
        If the file does not have a valid extension, raises a ValueError.

        Raises:
            ValueError: If the file does not have a valid NIfTI extension.

        Returns:
            str: The filename without the extension.
        """

        basename = os.path.basename(self.fname_nii)
        if basename.endswith('.nii.gz'):
            file_name = basename[:-7]  # Remove .nii.gz
        elif basename.endswith('.nii'):
            file_name = basename[:-4]
        else:
            raise ValueError("File does not have a valid NIfTI extension (.nii or .nii.gz)")

        return file_name

    @safe_getter(default_value=None)
    def get_path_nii(self) -> str:
        """Gets the path_nii of the Nifti file

        Returns:
            str: path_nii of the file (absolute path)
        """
        path_nii = os.path.dirname(self.fname_nii)

        # For files in current directory, return current working directory
        if not path_nii:
            path_nii = os.getcwd()

        return path_nii

    @safe_getter(default_value=None)
    def get_json_info(self, key: str, required: bool = False) -> any:
        """ Get a specific key from the JSON file.

        Args:
            key (str): The key to retrieve from the JSON file.
            required (bool): If True, raises KeyError when key not found. If False, returns None.

        Returns:
            any: The value associated with the key in the JSON file, or None if not found and required=False.

        Raises:
            KeyError: If the key is not found and required=True.
        """
        if self.json is not None and key in self.json:
            return self.json[key]
        elif required:
            raise KeyError(f"Key '{key}' not found in JSON file.")
        else:
            raise Warning(f"Key '{key}' not found in JSON file. Returning None.")

    @safe_getter(default_value=None)
    def get_isocenter(self) -> np.ndarray:
        """ Get the isocenter location in RAS coordinates from the json file.

        The patient position is used to infer the table position in the patient coordinate system.
        When the table is at (0,0,0), the origin is at the isocenter. We can therefore infer
        the isocenter as -table_position when the table_position is in RAS coordinates.

        Args:
            json_data (dict): Dictionary containing the BIDS sidecar information

        Returns:
            numpy.ndarray: Isocenter location in RAS coordinates
        """
        table_position = self.get_json_info('TablePosition')

        patient_position = self.get_json_info('PatientPosition')

        table_position = np.array(table_position)

        # Define coordinate transformations for each patient position
        position_transforms = {
            'HFS': [0, 1, 2],      # x=x, y=y, z=z
            'HFP': [0, 1, 2],      # x=-x, y=-y, z=z
            'FFS': [0, 1, 2],      # x=-x, y=y, z=-z
            'FFP': [0, 1, 2],      # x=x, y=-y, z=-z
            'LFP': [2, 1, 0],      # x=-z, y=-y, z=-x
            'LFS': [2, 1, 0],      # x=-z, y=y, z=x
            'RFP': [2, 1, 0],      # x=z, y=-y, z=x
            'RFS': [2, 1, 0],      # x=z, y=y, z=-x
            'HFDR': [1, 0, 2],     # x=-y, y=x, z=z
            'HFDL': [1, 0, 2],     # x=y, y=-x, z=z
            'FFDR': [1, 0, 2],     # x=-y, y=-x, z=-z
            'FFDL': [1, 0, 2],     # x=y, y=x, z=-z
            'AFDR': [1, 2, 0],     # x=-y, y=z, z=-x
            'AFDL': [1, 2, 0],     # x=y, y=z, z=x
            'PFDR': [1, 2, 0],     # x=-y, y=-z, z=x
            'PFDL': [1, 2, 0],     # x=y, y=-z, z=-x
        }

        # Define sign patterns for each patient position
        position_signs = {
            'HFS': [1, 1, 1],      'HFP': [-1, -1, 1],
            'FFS': [-1, 1, -1],    'FFP': [1, -1, -1],
            'LFP': [-1, -1, -1],   'LFS': [-1, 1, 1],
            'RFP': [1, -1, 1],     'RFS': [1, 1, -1],
            'HFDR': [-1, 1, 1],    'HFDL': [1, -1, 1],
            'FFDR': [-1, -1, -1],  'FFDL': [1, 1, -1],
            'AFDR': [-1, 1, -1],   'AFDL': [1, 1, 1],
            'PFDR': [-1, -1, 1],   'PFDL': [1, -1, -1],
        }

        if patient_position not in position_transforms:
            raise ValueError(f"Patient position {patient_position} not implemented")

        # Transform table position to RAS coordinates
        indices = position_transforms[patient_position]
        signs = position_signs[patient_position]

        table_position_ras = np.zeros(3)
        for i in range(3):
            table_position_ras[i] = signs[i] * table_position[indices[i]]

        # The isocenter is located at -table_position
        return -table_position_ras

    @safe_getter(default_value=None)
    def get_frequency(self) -> int | None:
        """ Get the imaging frequency from the JSON metadata.

        Returns:
            float: Imaging frequency in Hz, or None if not available.
        """
        frequency = self.get_json_info('ImagingFrequency', required=False)

        return int(frequency * 1e6) if frequency is not None else None

    def get_scanner_shim_settings(self, orders: list[int] = [0, 1, 2, 3]) -> dict:
        """ Get the scanner's shim settings using the BIDS tag ShimSetting and ImagingFrequency and returns it in a
            dictionary. 'orders' is used to check if the different orders are available in the metadata.

        Args:
            self (NiftiFile): The NiftiFile object containing the BIDS metadata.
            orders (list[int]): List of orders to check for shim settings. Default is [0, 1, 2, 3].

        Returns:
            dict: Dictionary containing the following keys: '0', '1' '2', '3'. The different orders are
                lists unless the different values could not be populated.
        """

        scanner_shim = {
            '0': [self.get_frequency()] if self.get_frequency() is not None else None,
            '1': None,
            '2': None,
            '3': None
        }

        # get_shim_orders
        shim_settings_list = self.get_json_info('ShimSetting')
        if shim_settings_list is not None:
            n_shim_values = len(shim_settings_list)
            if n_shim_values == 3:
                scanner_shim['1'] = shim_settings_list
            elif n_shim_values == 8:
                scanner_shim['1'] = shim_settings_list[:3]
                scanner_shim['2'] = shim_settings_list[3:]
            else:
                logger.warning(f"ShimSetting tag has an unsupported number of values: {n_shim_values}")
        else:
            logger.debug("ShimSetting tag is not available")

        # Check if the orders to shim are available in the metadata
        for order in orders:
            if scanner_shim.get(str(order)) is None:
                logger.debug(
                    f"Order {order} shim settings not available in the JSON metadata, constraints might not be "
                    f"respected.")

        return scanner_shim

    @safe_getter(default_value=None)
    def get_manufacturers_model_name(self) -> str:
        """ Get the manufacturer model from the JSON metadata.

        Returns:
            str: Manufacturer model name with spaces replaced by underscores, or None if not available.
        """
        model = self.get_json_info('ManufacturersModelName', required=False)
        return model.replace(" ", "_") if model is not None else None

    @safe_getter(default_value=False)
    def get_fat_sat_option(self) -> bool:
        """ Check if the NIfTI file has a Fat Saturation pulse.

        Returns:
            bool: True if Fat Saturation pulse is detected, False otherwise.
        """
        scan_options = self.get_json_info('ScanOptions', required=False)
        if scan_options is not None:
            if 'FS' in scan_options:
                logger.debug("Fat Saturation pulse detected")
                return True
        else:
            logger.warning("No ScanOptions found in the JSON metadata, assuming no Fat Saturation pulse")

        return False

    @safe_getter(default_value=False)
    def get_acquisition_times(self, when='slice-middle') -> np.ndarray:
        """ Return the acquisition timestamps from a json sidecar. This assumes BIDS convention.

        Args:
            when (str): When to get the acquisition time. Can be within {POSSIBLE_TIMINGS}.

        Returns:
            numpy.ndarray: Acquisition timestamps in ms (n_volumes x n_slices).
        """

        if when not in POSSIBLE_TIMINGS:
            raise ValueError(f"Invalid 'when' parameter. Must be within {POSSIBLE_TIMINGS}")

        # Get number of volumes
        n_volumes = self.header['dim'][4]
        n_slices = self.shape[2]

        deltat_volume = self.get_volume_tr()

        # Time the acquisition of data that this image started (ISO format)
        acq_start_time_ms = self.get_acquisition_start_time()

        # Start time for each volume [ms]
        volume_start_times = np.arange(acq_start_time_ms, (n_volumes * deltat_volume) + acq_start_time_ms, deltat_volume)

        if when == 'volume-start':
            return volume_start_times
        if when == 'volume-middle':
            return volume_start_times + (deltat_volume / 2)

        try:
            slice_timing_middle = self.get_phase_encode_0_crossings()
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

    @safe_getter(default_value=False)
    def get_slice_tr(self) -> float:
        """ Get the slice repetition time in milliseconds from the json sidecar.

        Returns:
            float: Slice repetition time in milliseconds.
        """
        excitation_tr = self.get_excitation_tr()

        pulse_sequence_details = self.get_json_info('PulseSequenceDetails')
        if pulse_sequence_details is None:
            raise ValueError("No PulseSequenceDetails name found in JSON file.")

        n_phase_encode_steps = self.get_n_acquired_phase_encode_lines()

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

    @safe_getter(default_value=False)
    def get_phase_encode_0_crossings(self) -> np.ndarray:
        """ Return the best guess of when the middle of k-space was acquired for each slice of 1 volume, relative to the
         start of the volume. This is implemented for Siemens, for 2D acquisitions and reads tags found in the JSON sidecar

        Returns:
            np.ndarray: Slice timing in ms (n_slices).
        """

        # Can be '2D'
        mr_acquisition_type = self.get_json_info('MRAcquisitionType')
        if mr_acquisition_type != '2D':
            # mr_acquisition_type is None or 3D
            raise NotImplementedError("MR acquisition type is not 2D.")

        # list containing the time at which each slice was acquired
        slice_timing_start = self.get_slice_timing()

        deltat_slice = self.get_slice_tr()

        # Error check
        volume_tr = self.get_volume_tr()

        n_slices = self.shape[2]
        if (deltat_slice * n_slices) * ERROR_MARGIN > volume_tr:
            ValueError("Slice timing of slices is longer than the volume timing.")

        # Get when the middle of k-space was acquired
        fourier = self.get_partial_fourier()

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

    @safe_getter(default_value=False)
    def get_slice_timing(self) -> np.ndarray:
        """ Get the slice timing from the json sidecar.

        Returns:
            numpy.ndarray: Slice timing in ms (n_slices).
        """
        # list containing the time at which each slice was acquired
        slice_timing_start = self.get_json_info('SliceTiming')
        n_slices = self.shape[2]
        if slice_timing_start is None:
            if n_slices == 1:
                # Slice timing information does not seem to be defined if there is only one slice
                slice_timing_start = [0.0]
            else:
                raise ValueError("No slice timing information found in JSON file.")
        else:
            slice_timing_start = np.array(slice_timing_start) * 1000  # [ms]

        return np.array(slice_timing_start)  # [ms]

    @safe_getter(default_value=False)
    def get_n_acquired_phase_encode_lines(self) -> int:
        """ Get the number of acquired phase encoding lines from the json sidecar.

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

        if self.get_json_info('Manufacturer') != 'Siemens':
            raise ValueError("This function is only implemented for Siemens scanners.")

        # Base resolution
        # PhaseEncodingSteps should include the calculations below, but I have noticed that it sometimes does not
        # phase_encode_steps = json_data.get('PhaseEncodingSteps')
        phase_encode_steps = self.get_json_info('AcquisitionMatrixPE')
        if phase_encode_steps is None:
            raise ValueError("AcquisitionMatrixPE not found in JSON sidecar.")
        # Convert to int if it was not None
        phase_encode_steps = int(phase_encode_steps)

        # Phase oversampling
        phase_over = self.get_json_info('PhaseOversampling')
        if phase_over is None:
            # If phase oversampling is not defined, assume it is 1 (no phase oversampling)
            phase_over = 1.0
        else:
            # PhaseOversampling is given as a percentage, we add 1 so it can be multiplied
            phase_over = float(phase_over) + 1.0

        # Partial Fourier
        partial_fourier = self.get_partial_fourier()

        n_phase_encode_steps = phase_encode_steps * partial_fourier * phase_over

        # Parallel acquisition reduction
        parallel_technique = self.get_json_info('ParallelAcquisitionTechnique')
        matrix_coil_mode = self.get_json_info('MatrixCoilMode')

        if parallel_technique is not None or matrix_coil_mode is not None:

            if parallel_technique == 'GRAPPA' or matrix_coil_mode == 'GRAPPA':
                parallel_reduction_factor_in_plane = self.get_json_info('ParallelReductionFactorInPlane')
                if parallel_reduction_factor_in_plane is None:
                    parallel_reduction_factor_in_plane = 1.0
                parallel_reduction_factor_in_plane = float(parallel_reduction_factor_in_plane)

                ref_lines_pe = self.get_json_info('RefLinesPE')
                if ref_lines_pe is None and parallel_reduction_factor_in_plane != 1.0:
                    raise ValueError("RefLinesPE not found in JSON sidecar.")
                ref_lines_pe = int(ref_lines_pe)

                n_phase_encode_steps = math.ceil((n_phase_encode_steps + ref_lines_pe) / parallel_reduction_factor_in_plane)
            else:
                NotImplementedError(f"{parallel_technique} parallel acquisition technique is not implemented yet.")

        ph_encode_steps_dcm2niix = self.get_json_info('PhaseEncodingSteps')
        if ph_encode_steps_dcm2niix is not None:
            ph_encode_steps_dcm2niix = int(ph_encode_steps_dcm2niix)
            if ph_encode_steps_dcm2niix != n_phase_encode_steps:
                logger.warning(f"PhaseEncodingSteps in JSON sidecar ({ph_encode_steps_dcm2niix}) does not match "
                               f"calculated value ({n_phase_encode_steps}). This is a bug in Shimming Toolbox or in "
                               f"dcm2niix. Using {n_phase_encode_steps} phase encoding steps (from ST).")

        return n_phase_encode_steps

    @safe_getter(default_value=False)
    def get_partial_fourier(self) -> float:
        """ Get the partial fourier value from the json sidecar.

        Returns:
            float: Partial Fourier value between 0 and 1. If no partial fourier is defined, returns 1.0.
        """
        manufacturer = self.get_json_info('Manufacturer')
        if manufacturer == 'Siemens':
            partial_fourier = self.get_json_info('PartialFourier')
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

    @safe_getter(default_value=False)
    def get_excitation_tr(self) -> float:
        """ Get the slice excitation repetition time in milliseconds from the json sidecar.

        Returns:
            float: Slice excitation repetition time in milliseconds.
        """

        # If RepetitionTimeExcitation is not defined, then the RepetitionTime is the time between 2 RF pulses of the
        # same slice
        if self.get_json_info('RepetitionTimeExcitation') is None:
            excitation_tr = self.get_json_info('RepetitionTime')
            if excitation_tr is None:
                raise ValueError("RepetitionTimeExcitation nor RepetitionTime is defined in the JSON sidecar. "
                                 "Can't figure out excitation TR")
        else:
            excitation_tr = self.get_json_info('RepetitionTimeExcitation')

        if excitation_tr is not None:
            excitation_tr = float(excitation_tr) * 1000  # [ms]

        return excitation_tr

    @safe_getter(default_value=False)
    def get_volume_tr(self) -> float:
        """ Get the volume repetition time in milliseconds from the json sidecar.

        Returns:
            float: Volume repetition time in milliseconds.
        """

        # If both RepetitionTimeExcitation and RepetitionTime are defined, then RepetitionTime is the volume TR and
        # RepetitionTimeExcitation is the time between 2 RF pulses of the same slice.
        # If RepetitionTimeExcitation is not defined, RepetitionTime is the time between 2 RF pulses of the same slice and
        # AcquisitionDuration is the time to acquire a single volume (i.e.: volume TR).
        if self.get_json_info('RepetitionTimeExcitation') is None:
            acq_duration = self.get_json_info('AcquisitionDuration')
            if acq_duration is None:
                # deltat_slice = get_slice_tr(json_data)
                # n_slices = nii_data.shape[2]
                # deltat_volume = deltat_slice * n_slices

                raise ValueError("RepetitionTimeExcitation nor AcquisitionDuration is defined in the JSON "
                                 "sidecar. Can't compute volume repetition time.")
            else:
                # If AcquisitionDuration is defined and RepetitionTimeExcitation is not,
                # then AcquisitionDuration is the time to acquire a single volume
                deltat_volume = float(acq_duration) * 1000
        else:
            # If RepetitionTimeExcitation is defined, RepetitionTime is the volume TR
            tr = self.get_json_info('RepetitionTime')
            if tr is None:
                raise ValueError("RepetitionTime is not defined in the JSON sidecar.")
            deltat_volume = float(tr) * 1000  # [ms]

        if deltat_volume is None:
            raise ValueError("Can't compute volume repetition time. RepetitionTimeExcitation, AcquisitionDuration and/or "
                             "RepetitionTime can't be used to determine volume TR.")

        return deltat_volume

    @safe_getter(default_value=False)
    def get_acquisition_duration(self) -> float:
        """ Compute the acquisition duration from the nifti data and the json sidecar.

        Returns:
            float: Acquisition duration in milliseconds.
        """

        if self.ndim == 4:
            n_volumes = self.shape[3]
        else:
            # If the data is not 4D, then there is no time dimension, so there is only a single volume
            n_volumes = 1

        deltat_volume = self.get_volume_tr()

        return n_volumes * deltat_volume  # [ms]

    @safe_getter(default_value=False)
    def get_acquisition_start_time(self) -> float:
        """
        Get the acquisition start time in milliseconds past midnight from the json sidecar.

        Returns:
            float: Acquisition start time in milliseconds past midnight.

        """
        acq_start_time_iso = self.get_json_info('AcquisitionTime')
        if acq_start_time_iso is None:
            raise ValueError("Acquisition time not found in json sidecar.")
        # todo: dummy scans?
        acq_start_time_ms = iso_times_to_ms(np.array([acq_start_time_iso]))[0]  # [ms]
        return acq_start_time_ms

    @safe_getter(default_value=False)
    def get_acquisition_stop_time(self) -> float:
        """
        Get the acquisition stop time in milliseconds past midnight from the json sidecar.

        Returns:
            float: Acquisition stop time in milliseconds past midnight.

        """

        acq_start_time = self.get_acquisition_start_time()
        acq_duration = self.get_acquisition_duration()
        acq_stop_time = acq_start_time + acq_duration

        # If the acquisition stop time is greater than 24 hours, then it is the next day
        return acq_stop_time % (SECONDS_IN_A_DAY * 1000)


# TODO: Implement NiftiCoilProfile class
class NiftiCoilProfile(NiftiFile):
    """NiftiCoilProfile is a subclass of NiftiFile that represents a NIfTI coil profile file.

    It inherits all methods and properties from NiftiFile and can be used to handle coil profile files specifically.
    """

    def __init__(self, fname_nii: str) -> None:
        super().__init__(fname_nii)
