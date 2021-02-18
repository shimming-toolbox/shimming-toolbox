#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Wrapper to different unwrapping algorithms. """

import numpy as np
import errno

from shimmingtoolbox.unwrap.prelude import prelude
from shimmingtoolbox.language import English as notice

def unwrap_phase(phase, affine, unwrapper='prelude', mag=None, mask=None, threshold=None):
    """ Calls different unwrapping algorithms according to the specified `unwrapper` parameter. The function also
    allows to call the different unwrappers with more flexibility regarding input shape.

    Args:
        phase (numpy.ndarray): 2D, 3D or 4D radian values [-pi to pi] to perform phase unwrapping.
                               Supported shapes: [x, y], [x, y, z] or [x, y, z, t].
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be acquired by :
            nii = nib.load("nii_path")
            affine = nii.affine
        unwrapper (str, optional): Unwrapper algorithm name. Possible values: ``prelude``.
        mag (numpy.ndarray): 2D, 3D or 4D magnitude data corresponding to phase data. Shape must be the same as
                     ``phase``.
        mask (numpy.ndarray): numpy array of booleans with shape of ``phase`` to mask during phase unwrapping.
        threshold (float): Prelude parameter, see prelude for more detail.

    Returns:
        numpy.ndarray: Unwrapped phase image.
    """

    if unwrapper == 'prelude':
        mag4d = None
        mask4d = None
        if phase.ndim == 2:
            phase4d = np.expand_dims(phase, -1)
            if mag is not None:
                mag4d = np.expand_dims(mag, -1)
            if mask is not None:
                mask4d = np.expand_dims(mask, -1)

            mag = mag4d
            mask = mask4d

            phase3d_unwrapped = prelude(phase4d, affine, mag=mag, mask=mask, threshold=threshold)

            phase_unwrapped = phase3d_unwrapped[..., 0]

        elif phase.ndim == 3:
            phase_unwrapped = prelude(phase, affine, mag=mag, mask=mask, threshold=threshold)

        elif phase.ndim == 4:
            phase_unwrapped = np.zeros_like(phase)
            for i_t in range(phase.shape[3]):
                mask3d = None
                mag3d = None

                phase3d = phase[..., i_t]
                if mag is not None:
                    mag3d = mag[..., i_t]
                if mask is not None:
                    mask3d = mask[..., i_t]

                mask_input = mask3d
                mag_input = mag3d

                phase_unwrapped[..., i_t] = prelude(phase3d, affine, mag=mag_input, mask=mask_input, threshold=threshold)

        else:
            raise ValueError( errno.ENODATA, notice._unsupported_phase )

    else:
        # TODO: Assert that the unwrap's name is caught in the standard error
        raise NotImplementedError(errno.ENODATA, notice._unimplemented_unwrap )

    return phase_unwrapped
