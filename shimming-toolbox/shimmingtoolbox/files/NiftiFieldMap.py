import logging
import nibabel as nib
import numpy as np
import os
import copy
import math
from .NiftiFile import NiftiFile
from shimmingtoolbox.shim.shim_utils import extend_slice

logger = logging.getLogger(__name__)

class NiftiFieldMap(NiftiFile):
    """NiftiFieldMap is a subclass of NiftiFile that represents a NIfTI field map file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle field map files specifically.
    """
    def __init__(self, fname_nii: str, dilation_kernel_size, json:dict = None, path_output: str = None, isRealtime: bool = False) -> None:
        super().__init__(fname_nii, json=json, path_output=path_output)
        self.dilation_kernel_size = dilation_kernel_size
        self.isRealtime = isRealtime
        self.location = None
        self.extended_nii = self.extend_field_map(dilation_kernel_size)
        self.extended_data = self.extended_nii.get_fdata()
        self.extended_affine = self.extended_nii.affine
        self.extended_shape = self.extended_data.shape
    
    def set_nii(self, nii: nib.Nifti1Image) -> None:
        """ Set the NIfTI image and update the data, affine, and shape attributes.
        
        Args:
            nii (nib.Nifti1Image): The NIfTI image to set.
        """
        super().set_nii(nii)
        self.extend_field_map(self.dilation_kernel_size)
        
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
        if self.ndim != 3 and not self.isRealtime:
            if self.ndim == 2:
                super().set_nii(nib.Nifti1Image(self.data[..., np.newaxis], self.affine,
                                                header=self.header))
                extended_nii = self.extend_fmap_to_kernel_size(dilation_kernel_size)
                self.extended = True
            else:
                raise ValueError("Fieldmap must be 2d or 3d")
        else:
            if self.isRealtime and self.ndim != 4:
                raise ValueError("Fieldmap must be 4d for realtime processing")
            
            for i_axis in range(3):
                if self.shape[i_axis] < dilation_kernel_size:
                    self.extended = True
                    break
                
            if self.extended:
                if self.isRealtime:
                    extended_nii, location = self.extend_fmap_to_kernel_size(dilation_kernel_size, ret_location=True)
                    self.location = location
                else:
                    extended_nii = self.extend_fmap_to_kernel_size(dilation_kernel_size)
                    
                self.extended = True
                
            if logger.level <= getattr(logging, 'DEBUG') and self.extended:
                logger.debug(f"Field map shape: {self.shape}, "
                             f"Extended shape: {extended_nii.shape if self.extended else self.shape}")
                nib.save(extended_nii, os.path.join(self.path_output, f"{self.filename}_extended.nii.gz"))

        return extended_nii if self.extended else self.nii
    
    def extend_fmap_to_kernel_size(self, dilation_kernel_size, ret_location=False):
        """
        Load the fmap and expand its dimensions to the kernel size

        Args:
            dilation_kernel_size: Size of the kernel
            ret_location (bool): If True, return the location of the original data in the new data
        Returns:
            nib.Nifti1Image: Nibabel object of the loaded and extended fieldmap
        """
        fieldmap_shape = self.shape[:3]

        # Extend the dimensions where the kernel is bigger than the number of voxels
        tmp_nii = copy.deepcopy(self.nii)
        location = np.ones(self.shape)
        for i_axis in range(len(fieldmap_shape)):
            # If there are less voxels than the kernel size, extend in that axis
            if fieldmap_shape[i_axis] < dilation_kernel_size:
                diff = float(dilation_kernel_size - fieldmap_shape[i_axis])
                n_slices_to_extend = math.ceil(diff / 2)
                tmp_nii, location = extend_slice(tmp_nii, n_slices=n_slices_to_extend, axis=i_axis, location=location)

        nii_fmap = tmp_nii

        # If DEBUG, save the extended fieldmap
        if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
            fname_new_fmap = os.path.join(self.path_output, 'tmp_extended_fmap.nii.gz')
            nib.save(nii_fmap, fname_new_fmap)
            logger.debug(f"Extended fmap, saved the new fieldmap here: {fname_new_fmap}")

        if ret_location:
            return nii_fmap, location.astype(bool)

        return nii_fmap