from __future__ import annotations
import logging
import nibabel as nib
import os
import numpy as np
import json
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_getter(default_value=None):
    """Decorator that catches errors in getter functions and returns a default value."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                # terminate the program if the error is critical
                if isinstance(e, (NameError, ValueError)):
                    raise e
                return default_value
        return wrapper
    return decorator

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')
DEFAULT_SUFFIX = '_saved.nii.gz'

class NiftiFile:
    def __init__(self, fname_nii: str) -> None:
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
        self.json: dict | None = self.load_json()
        self.header = self.nii.header
        self.affine = self.nii.affine
        self.shape = self.data.shape
        self.ndim = self.data.ndim

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
        data = nii.get_fdata()
        
        return nii, data
    
    def load_json(self):
        """ Load the JSON file corresponding to the NIfTI file.
        The JSON file is expected to be in the same directory as the NIfTI file
        and have the same base name.

        Args:
            None

        Returns:
            dict: The content of the JSON file if found, otherwise None.
        """
        json_path = self.get_json()
        if json_path is not None:
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            return None
        
    def save(self, output_path: str | None = None) -> None:
        """ Save the NIfTI file to a specified path.
        If no output path is provided, it saves the file in the same directory with a default name.

        Args:
            output_path (str, optional): The path where the NIfTI file should be saved.
                If None, it saves the file in the same directory with a default name.
                
        Raises:
            ValueError: If the output path is not a valid directory.
            
        Returns:
            None: The function saves the NIfTI file to the specified path.
        """
        if output_path is None:
            output_path = os.path.join(self.path_nii, f"{self.filename}{DEFAULT_SUFFIX}")
            logger.warning(f"No output path provided. Saving as {output_path}")
        
        output_dir = os.path.path_nii(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif not os.path.isdir(output_dir):
            raise ValueError(f"Output path {output_path} is not a valid directory.")
            
        nib.save(self.nii, output_path)
    
    @safe_getter(default_value=None)
    def get_json(self):
        """ Find the corresponding JSON file for the NIfTI file.
        The JSON file is expected to be in the same directory as the NIfTI file
        and have the same base name.

        Args:
            None

        Returns:
            str: The path to the JSON file if found, otherwise None.
        """
        json_path = os.path.join(self.path_nii, self.filename + ".json")
        if os.path.exists(json_path):
            return json_path
        else:
            return None
    
    @safe_getter(default_value=None)
    def get_filename(self):
        """ Get the filename without the extension from the NIfTI file path.
        Verifies that the file has a valid NIfTI extension (.nii or .nii.gz).
        If the file does not have a valid extension, raises a ValueError.
        
        Args:
            None

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
        path_nii = os.path.path_nii(self.fname_nii)
        
        # For files in current directory, return current working directory
        if not path_nii:
            path_nii = os.getcwd()
        
        return path_nii
    
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
        self.data = nii.get_fdata()
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
    def get_json_info(self, key: str) -> any:
        """ Get a specific key from the JSON file.

        Args:
            key (str): The key to retrieve from the JSON file.

        Returns:
            any: The value associated with the key in the JSON file, or None if the key does not exist.
        """
        if self.json is not None and key in self.json:
            return self.json[key]
        else:
            raise KeyError(f"Key '{key}' not found in JSON file.")
        
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
        if table_position is None:
            raise ValueError("Table position not found in json sidecar.")

        patient_position = self.get_json_info('PatientPosition')
        if patient_position is None:
            raise ValueError("Patient position not found in json sidecar.")

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
