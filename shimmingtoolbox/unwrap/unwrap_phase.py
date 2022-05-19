#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Wrapper to different unwrapping algorithms. """

import numpy as np
import logging
import nibabel as nib

from shimmingtoolbox.unwrap.prelude import prelude

logger = logging.getLogger(__name__)


def unwrap_phase(nii_phase_wrapped, unwrapper='prelude', mag=None, mask=None, threshold=None, fname_save_mask=None):
    """ Calls different unwrapping algorithms according to the specified `unwrapper` parameter. The function also
    allows to call the different unwrappers with more flexibility regarding input shape.

    Args:
        nii_phase_wrapped (nib.Nifti1Image): 2D, 3D or 4D radian values [-pi to pi] to perform phase unwrapping.
                                             Supported shapes: [x, y], [x, y, z] or [x, y, z, t].
        unwrapper (str, optional): Unwrapper algorithm name. Possible values: ``prelude``.
        mag (numpy.ndarray): 2D, 3D or 4D magnitude data corresponding to phase data. Shape must be the same as
                     ``phase``.
        mask (numpy.ndarray): numpy array of booleans with shape of ``phase`` to mask during phase unwrapping.
        threshold (float): Prelude parameter, see prelude for more detail.
        fname_save_mask (str): Filename of the mask calculated by the unwrapper

    Returns:
        numpy.ndarray: Unwrapped phase image.
    """

    phase = nii_phase_wrapped.get_fdata()

    if unwrapper == 'prelude':
        mag2d = None
        mask2d = None
        if phase.ndim == 2:
            phase2d = np.expand_dims(phase, -1)
            if mag is not None:
                mag2d = np.expand_dims(mag, -1)
            if mask is not None:
                mask2d = np.expand_dims(mask, -1)

            mag = mag2d
            mask = mask2d
            nii_2d = nib.Nifti1Image(phase2d, nii_phase_wrapped.affine, header=nii_phase_wrapped.header)

            logger.info(f"Unwrapping 1 volume")
            phase3d_unwrapped = prelude(nii_2d, mag=mag, mask=mask, threshold=threshold,
                                        fname_save_mask=fname_save_mask)

            phase_unwrapped = phase3d_unwrapped[..., 0]

        elif phase.ndim == 3:
            logger.info("Unwrapping 1 volume")
            phase_unwrapped = prelude(nii_phase_wrapped, mag=mag, mask=mask, threshold=threshold,
                                      fname_save_mask=fname_save_mask)

        elif phase.ndim == 4:

            logger.info(f"Unwrapping {phase.shape[3]} volumes")
            phase_unwrapped = np.zeros_like(phase)
            n_t = phase.shape[3]
            for i_t in range(n_t):
                mask4d = None
                mag4d = None

                phase4d = phase[..., i_t]
                nii_4d = nib.Nifti1Image(phase4d, nii_phase_wrapped.affine, header=nii_phase_wrapped.header)

                if mag is not None:
                    mag4d = mag[..., i_t]
                if mask is not None:
                    mask4d = mask[..., i_t]

                mask_input = mask4d
                mag_input = mag4d

                phase_unwrapped[..., i_t] = prelude(nii_4d, mag=mag_input, mask=mask_input, threshold=threshold)

                # If it's the first volume, call it using save mask to save the mask only once
                if i_t == 0:
                    phase_unwrapped[..., i_t] = prelude(nii_4d, mag=mag_input, mask=mask_input, threshold=threshold,
                                                        fname_save_mask=fname_save_mask)

        else:
            raise ValueError("Shape of input phase is not supported.")

    else:
        raise NotImplementedError(f'The unwrap function {unwrapper} is not implemented.')

    return phase_unwrapped
