import logging
import nibabel as nib
import numpy as np
from .NiftiFile import NiftiFile, safe_getter

logger = logging.getLogger(__name__)

class NiftiAnatomical(NiftiFile):
    """NiftiAnatomical is a subclass of NiftiFile that represents a NIfTI anatomical file.
    
    It inherits all methods and properties from NiftiFile and can be used to handle anatomical files specifically.
    """
    def __init__(self, fname_nii: str, path_output: str = None) -> None:
        super().__init__(fname_nii)
        self.check_dimensions()
        self.temporal_average()
        
    def temporal_average(self):
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
            logger.debug("No ScanOptions found in the JSON metadata, assuming no Fat Saturation pulse")
        
        return False