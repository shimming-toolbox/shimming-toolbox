import nibabel as nib
import os
import logging
from __future__ import annotations
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
                print(f"Error in {func.__name__}: {e}")
                # terminate the program if the error is critical
                if isinstance(e, (NameError, ValueError)):
                    raise e
                return default_value
        return wrapper
    return decorator


class NiftiFile:
    def __init__(self, path_nii):
        self.path_nii = path_nii
        self.dirname = self.get_dirname()
        self.nii, self.data = self.load_nii()
        self.filename = self.get_filename()
        self.json = self.load_json()
        self.header = self.nii.header
        self.affine = self.nii.affine

    def load_nii(self):
        """ Load a NIfTI file and return the NIfTI object and its data.
        Args:
            path_nii (str): Path to the NIfTI file.

        Raises:
            ValueError: If the provided path does not exist or is not a valid NIfTI file.
            
        Returns:
            nib.Nifti1Image: The loaded NIfTI image object.
            numpy.ndarray: The data contained in the NIfTI file.
        """
        if not os.path.exists(self.path_nii):
            raise ValueError("Not an existing NIFTI path")
        nii = nib.load(self.path_nii)
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
                import json
                return json.load(f)
        else:
            return None
        
    def save(self, output_path=None):
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
            output_path = os.path.join(self.dirname, self.filename + "_saved.nii.gz")
            # Warning to the user that the file will be saved with a default name
            logger.info(f"Warning: No output path provided. Saving as {output_path}")
        if not os.path.isdir(os.path.dirname(output_path)):
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
        json_path = os.path.join(self.dirname, self.filename + ".json")
        if os.path.exists(json_path):
            return json_path
        else:
            return None
    
    @safe_getter(default_value="unknown")
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
        
        basename = os.path.basename(self.path_nii)
        if basename.endswith('.nii.gz'):
            file_name = basename[:-7]  # Remove .nii.gz
        elif basename.endswith('.nii'):
            file_name = basename[:-4]
        else:
            raise ValueError("File does not have a valid NIfTI extension (.nii or .nii.gz)")
        
        return file_name  
    
    @safe_getter(default_value="")
    def get_dirname(self):
        """Gets the dirname of the Nifti file

        Raises:
            ValueError: If the dirname doesn't exist

        Returns:
            str: driname of the file
        """
        dirname = os.path.dirname(self.path_nii)
        
        if not dirname:
            raise ValueError("Dirname is empty for file " + self.path_nii)
        
        return dirname