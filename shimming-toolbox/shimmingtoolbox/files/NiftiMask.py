#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import nibabel as nib
import numpy as np
import os

from .NiftiFile import NiftiFile
from .NiftiTarget import NiftiTarget
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.coils.coordinates import resample_from_to

logger = logging.getLogger(__name__)


class NiftiMask(NiftiFile):
    """NiftiMask is a subclass of NiftiFile that represents a NIfTI mask file.

    It inherits all methods and properties from NiftiFile and can be used to handle mask files specifically.
    """
    def __init__(self, fname_nii: str, json:dict = None, path_output: str = None) -> None:
        super().__init__(fname_nii, json=json, path_output=path_output, json_needed=False)

    def set_nii(self, nii: nib.Nifti1Image, nif_target: NiftiTarget) -> None:
        """ Set the NIfTI image and load the mask on the target image.

        Args:
            nii (nib.Nifti1Image): The NIfTI image to set, which should be a mask.
            nif_target (NiftiTarget): The target image to resample the mask on.
        """
        super().set_nii(nii)
        self.load_mask(nif_target)

    def load_mask(self, nif_target: NiftiTarget):
        """ Load a mask and resample it on the target image.

        Args:
            nif_target (NiftiTarget): The target image to resample the mask on.

        Raises:
            ValueError: If the mask is not in 3D or 4D.
        """
        if self.ndim == 3:
            nii_mask_target = self.nii
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
            nii_mask_target = nib.Nifti1Image(tmp_3d.astype(int), self.affine, header=self.header)
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_target, os.path.join(self.path_output, "fig_3d_mask.nii.gz"))
        else:
            raise ValueError("Mask must be in 3d or 4d")


        # Check if the mask needs to be resampled
        if not np.all(nii_mask_target.shape == nif_target.shape) or not np.all(nii_mask_target.affine == nif_target.affine):
            # Resample the mask on the target
            logger.debug("Resampling mask on the target")
            nii_mask_target = resample_from_to(nii_mask_target, nif_target, order=1, mode='grid-constant')
            # Save the resampled mask
            if logger.level <= getattr(logging, 'DEBUG') and self.path_output is not None:
                nib.save(nii_mask_target, os.path.join(self.path_output, "mask_static_resampled_on_target.nii.gz"))

        super().set_nii(nii_mask_target)
