from __future__ import annotations
import logging
import nibabel as nib
import os
import numpy as np
import json
import copy
from functools import wraps
from shimmingtoolbox.shim.sequencer import extend_fmap_to_kernel_size
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import resample_from_to

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
                logger.warning(f"{func.__name__}: {e}")
                # terminate the program if the error is critical
                if isinstance(e, (KeyError, NameError, ValueError)):
                    raise e
                return default_value
        return wrapper
    return decorator

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')
DEFAULT_SUFFIX = '_saved.nii.gz'

class NiftiFile:
    def __init__(self, fname_nii: str, path_output: str = None):
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
        self.path_output = path_output if path_output else os.getcwd()
        
    def __eq__(self, other: nib.Nifti1Image) -> None:
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
        
    def save(self) -> None:
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
        if self.path_output is None:
            output_path = os.path.join(self.path_nii, f"{self.filename}{DEFAULT_SUFFIX}")
            logger.warning(f"No output path provided. Saving as {output_path}")
        else:
            output_path = os.path.join(self.path_output, f"{self.filename}{DEFAULT_SUFFIX}")
            logger.info(f"Saving NIfTI file to {output_path}")
            
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        elif not os.path.isdir(self.path_output):
            raise ValueError(f"Output path {output_path} is not a valid directory.")
            
        nib.save(self.nii, output_path)
    
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

        Returns:
            dict: Dictionary containing the following keys: '0', '1' '2', '3'. The different orders are
                lists unless the different values could not be populated.
        """

        scanner_shim = {
            '0': None,
            '1': None,
            '2': None,
            '3': None
        }
        # get_imaging_frequency
        scanner_shim['0'] = [self.get_frequency()] if self.get_frequency() is not None else None

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
                logger.debug(f"Order {order} shim settings not available in the JSON metadata, constraints might not be "
                            f"respected.")

        return scanner_shim

    @safe_getter(default_value=None)
    def get_manufacturer_model_name(self) -> str:
        """ Get the manufacturer model from the JSON metadata.

        Returns:
            str: Manufacturer model name with spaces replaced by underscores, or None if not available.
        """
        model = self.get_json_info('ManufacturerModelName', required=False)
        return model.replace(" ", "_") if model is not None else None


class NiftiFieldMap(NiftiFile):
    """NiftiFieldMap is a subclass of NiftiFile that represents a NIfTI field map file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle field map files specifically.
    """
    def __init__(self, fname_nii: str, dilation_kernel_size, path_output: str = None) -> None:
        super().__init__(fname_nii, path_output)
        self.extended_nii = self.extend_field_map(dilation_kernel_size)
        self.extended_data = self.extended_nii.get_fdata()
        self.extended_affine = self.extended_nii.affine
        self.extended_shape = self.extended_data.shape
        
    def extend_field_map(self, dilation_kernel_size: int) -> None:
        """ Extend the field map to match the dilation kernel size.
        This method checks the dimensions of the field map and extends it if necessary.
        Args:
            dilation_kernel_size (int): The size of the dilation kernel to extend the field map to.
        Raises:
            ValueError: If the field map is not 2D or 3D.
        Returns:
            numpy.array : The extended NIfTI image if the field map was extended, otherwise the original NIfTI image.
        """
        self.extended = False
        if self.ndim != 3:
            if self.ndim == 2:
                extended_nii = nib.Nifti1Image(self.data[..., np.newaxis], self.affine,
                                                header=self.header)
                self.extended = True
            else:
                raise ValueError("Fieldmap must be 2d or 3d")
        else:
            extending = False
            for i_axis in range(3):
                if self.shape[i_axis] < dilation_kernel_size:
                    self.extended = True
            if self.extended:
                extended_nii = extend_fmap_to_kernel_size(self.nii, dilation_kernel_size)
                
            if logger.level <= getattr(logging, 'DEBUG') and self.extended:
                logger.debug(f"Field map shape: {self.shape}, "
                             f"Extended shape: {extended_nii.shape if self.extended else self.shape}")
                nib.save(extended_nii, os.path.join(self.path_output, f"{self.filename}_extended.nii.gz"))

        return extended_nii if extending else self.nii


class NiftiAnatomical(NiftiFile):
    """NiftiAnatomical is a subclass of NiftiFile that represents a NIfTI anatomical file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle anatomical files specifically.
    """
    def __init__(self, fname_nii: str, path_output: str = None) -> None:
        super().__init__(fname_nii)
        self.check_dimensions()
        self.average_field_map()
        
    def average_field_map(self):
        if self.ndim == 3:
            pass
        elif self.ndim == 4:
            logger.info("Target anatomical is 4d, taking the average and converting to 3d")
            self.set_nii(nib.Nifti1Image(np.mean(self.data, axis=3), self.affine, header=self.header))
        else:
            raise ValueError("Target anatomical image must be in 3d or 4d") 
        
    def check_dimensions(self):
        dim_info = self.header.get_dim_info()
        
        if dim_info[2] is None:
            logger.warning("The slice encoding direction is not specified in the NIfTI header, Shimming Toolbox will "
                        "assume it is in the third dimension.")
        else:
            if dim_info[2] != 2:
                # # Reorient nifti so that the slice is the last dim
                # anat = nii_anat.get_fdata()
                # # TODO: find index of dim_info
                # index_in = 0
                # index_out = 2
                #
                # # Swap axis in the array
                # anat = np.swapaxes(anat, index_in, index_out)
                #
                # # Affine must change
                # affine = copy.deepcopy(nii_anat.affine)
                # affine[:, index_in] = nii_anat.affine[:, index_out]
                # affine[:, index_out] = nii_anat.affine[:, index_in]
                # affine[index_out, 3] = nii_anat.affine[index_in, 3]
                # affine[index_in, 3] = nii_anat.affine[index_out, 3]
                #
                # nii_reorient = nib.Nifti1Image(anat, affine, header=nii_anat.header)
                # nib.save(nii_reorient, os.path.join(path_output, 'anat_reorient.nii.gz'))

                # Slice must be the 3rd dimension of the file
                # TODO: Reorient nifti so that the slice is the 3rd dim
                raise RuntimeError("Slice encode direction must be the 3rd dimension of the NIfTI file.") 
    
            
    class NiftiMask(NiftiFile):
        """NiftiMask is a subclass of NiftiFile that represents a NIfTI mask file.
        
        It inherits all methods and properties from NiftiFile and can be used to handle mask files specifically.
        """
        def __init__(self, fname_nii: str, path_output: str = None) -> None:
            super().__init__(fname_nii, path_output)


class NiftiMask(NiftiFile):
    """NiftiMask is a subclass of NiftiFile that represents a NIfTI mask file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle mask files specifically.
    """
    def __init__(self, fname_nii: str, path_output: str = None) -> None:
        super().__init__(fname_nii, path_output)
    
    def load_mask(self, nii_anat: NiftiAnatomical):
        """ Load a mask and resample it on the target anatomical image.

        Args:
            nii_anat (NiftiAnatomical): The target anatomical image to resample the mask on.

        Raises:
            ValueError: If the mask is not in 3D or 4D.
        """
        if self.ndim == 3:
            pass
        elif self.ndim == 4:
            logger.debug("Mask is 4d, converting to 3d")
            tmp_3d = np.zeros(self.shape[:3])
            n_vol = self.shape[-1]
            # Summing over 4th dimension making sure that the max value is 1
            for i_vol in range(self.shape[-1]):
                tmp_3d += (self.data[..., i_vol] / self.data[..., i_vol].max())
            # 80% of the volumes must contain the desired pixel to be included, this avoids having dead voxels in the
            # output mask
            tmp_3d = threshold(tmp_3d, thr=int(n_vol * 0.8))
            nii_mask_anat = nib.Nifti1Image(tmp_3d.astype(int), nii_mask_anat.affine,
                                            header=nii_mask_anat.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_anat, os.path.join(self.path_output, "fig_3d_mask.nii.gz"))
        else:
            raise ValueError("Mask must be in 3d or 4d")

        if not np.all(nii_mask_anat.shape == nii_anat.shape) or not np.all(
                nii_mask_anat.affine == nii_anat.affine):
            logger.debug("Resampling mask on the target anat")
            nii_mask_anat_soft = resample_from_to(nii_mask_anat, nii_anat.nii, order=1, mode='grid-constant')
            tmp_mask = nii_mask_anat_soft.get_fdata()
            # Change soft mask into binary mask
            tmp_mask = threshold(tmp_mask, thr=0.001, scaled_thr=True)
            nii_mask_anat = nib.Nifti1Image(tmp_mask, nii_mask_anat_soft.affine, header=nii_mask_anat_soft.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_anat, os.path.join(self.path_output, "mask_static_resampled_on_anat.nii.gz"))
        
        self.set_nii(nii_mask_anat)
    
# TODO: Implement NiftiCoilProfile class
class NiftiCoilProfile(NiftiFile):
    """NiftiCoilProfile is a subclass of NiftiFile that represents a NIfTI coil profile file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle coil profile files specifically.
    """
    def __init__(self, fname_nii: str) -> None:
        super().__init__(fname_nii)