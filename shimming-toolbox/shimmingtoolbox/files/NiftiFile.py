#!/usr/bin/python3
# -*- coding: utf-8 -*

from __future__ import annotations
import logging
import nibabel as nib
import os
import numpy as np
import json
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')
DEFAULT_SUFFIX = '_saved.nii.gz'


def safe_getter(default_value=None):
    """Decorator that catches errors in getter functions and returns a default value."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__}: {e}")
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

    def load_nii(self):
        """ Load a NIfTI file and return the NIfTI object and its data.
        Args:
            fname_nii (str): Path to the NIfTI file.

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
            None

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
                if len(fname.split('.')) == 0:
                    raise ValueError("File name must end with .nii or .nii.gz")
                else:
                    fname += DEFAULT_SUFFIX
            fname_output = os.path.join(self.path_output, fname)
        else:
            fname_output = os.path.join(self.path_output, f"{self.filename}{DEFAULT_SUFFIX}")

        logger.info(f"Saving NIfTI file to {fname_output}")

        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        elif not os.path.isdir(self.path_output):
            raise ValueError(f"Output path {fname_output} is not a valid directory.")

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
    def get_filename(self):
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
    def get_path_nii(self):
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
    def get_isocenter(self):
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
    def get_frequency(self):
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


# TODO: Implement NiftiCoilProfile class
class NiftiCoilProfile(NiftiFile):
    """NiftiCoilProfile is a subclass of NiftiFile that represents a NIfTI coil profile file.

    It inherits all methods and properties from NiftiFile and can be used to handle coil profile files specifically.
    """

    def __init__(self, fname_nii: str) -> None:
        super().__init__(fname_nii)
