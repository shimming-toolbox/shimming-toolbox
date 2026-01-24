#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import nibabel as nib
import numpy as np

from .NiftiFile import NiftiFile, safe_getter

logger = logging.getLogger(__name__)


class NiftiTarget(NiftiFile):
    """NiftiTarget is a subclass of NiftiFile that represents a NIfTI target image file.

    It inherits all methods and properties from NiftiFile and can be used to handle target image files specifically.
    """
    def __init__(self, fname_nii: str, json:dict = None, path_output: str = None) -> None:
        super().__init__(fname_nii, json=json, path_output=path_output)
        self.check_dimensions()
        self.make_3d()

    def set_nii(self, nii: nib.Nifti1Image) -> None:
        """ Set the NIfTI image and update the data, affine, and shape attributes.

        Args:
            nii (nib.Nifti1Image): The NIfTI image to set.
        """
        super().set_nii(nii)
        self.check_dimensions()
        self.make_3d()

    def make_3d(self) -> None:
        if self.ndim == 3:
            pass
        elif self.ndim == 4:
            logger.info("Target image is 4d, taking the average and converting to 3d")
            self.set_nii(nib.Nifti1Image(np.mean(self.data, axis=3), self.affine, header=self.header))
        else:
            raise ValueError("Target image must be in 3d or 4d")

    def check_dimensions(self) -> None:
        dim_info = self.header.get_dim_info()

        if dim_info[2] is None:
            logger.warning("The slice encoding direction is not specified in the NIfTI header, Shimming Toolbox will "
                        "assume it is in the third dimension.")
        else:
            if dim_info[2] != 2:
                # # Reorient nifti so that the slice is the last dim
                # target = nii_target.get_fdata()
                # # TODO: find index of dim_info
                # index_in = 0
                # index_out = 2
                #
                # # Swap axis in the array
                # target = np.swapaxes(target, index_in, index_out)
                #
                # # Affine must change
                # affine = copy.deepcopy(nii_target.affine)
                # affine[:, index_in] = nii_target.affine[:, index_out]
                # affine[:, index_out] = nii_target.affine[:, index_in]
                # affine[index_out, 3] = nii_target.affine[index_in, 3]
                # affine[index_in, 3] = nii_target.affine[index_out, 3]
                #
                # nii_reorient = nib.Nifti1Image(target, affine, header=nii_target.header)
                # nib.save(nii_reorient, os.path.join(path_output, 'target_reorient.nii.gz'))

                # Slice must be the 3rd dimension of the file
                # TODO: Reorient nifti so that the slice is the 3rd dim
                raise RuntimeError("Slice encode direction must be the 3rd dimension of the NIfTI file.")
