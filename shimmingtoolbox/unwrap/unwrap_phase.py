#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Wrapper to different unwrapping algorithms """

import numpy as np
from shimmingtoolbox.unwrap.prelude import prelude


def unwrap_phase(phase, mag, affine, unwrapper='prelude', mask=None, threshold=None):
    """ Calls different unwrapping algorithms according to the specified `unwrapper` parameter. The function also
    allows to call the different unwrappers with more flexibility regarding input shape.

    Args:
        phase (numpy.ndarray): 2D, 3D or 4D radian values [-pi to pi] to perform phase unwrapping.
                               Supported shapes: [x, y], [x, y, z] or [x, y, z, t]
        mag (numpy.ndarray): 2D, 3D or 4D magnitude data corresponding to phase data. Shape must be the same as
                             ``phase``
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be acquired by :
            nii = nib.load("nii_path")
            affine = nii.affine
        unwrapper (str, optional): Unwrapper algorithm name. Possible values: ``prelude``
        mask (numpy.ndarray): numpy array of booleans with shape of ``phase`` to mask during phase unwrapping.
        threshold (float): Prelude parameter, see prelude for more detail

    Returns:
        numpy.ndarray: Unwrapped phase image
    """

    if unwrapper == 'prelude':

        # Make sure phase is 4d
        if phase.ndim == 2:
            phase4d = np.expand_dims(np.expand_dims(phase, -1), -1)
            mag4d = np.expand_dims(np.expand_dims(mag, -1), -1)
            if mask is not None:
                mask4d = np.expand_dims(np.expand_dims(mask, -1), -1)
        elif phase.ndim == 3:
            phase4d = np.expand_dims(phase, -1)
            mag4d = np.expand_dims(mag, -1)
            if mask is not None:
                mask4d = np.expand_dims(mask, -1)
        elif phase.ndim == 4:
            phase4d = phase
            mag4d = mag
            if mask is not None:
                mask4d = mask
        else:
            raise RuntimeError("Shape of input phase is not supported")

        # Split along 4th dimension (time), run prelude for each instance and merge back
        phase4d_unwrapped = np.zeros_like(phase4d)
        for i_t in range(phase4d.shape[3]):
            if mask is not None:
                phase4d_unwrapped[..., i_t] = prelude(phase4d[..., i_t], mag4d[..., i_t], affine, mask=mask4d[..., i_t],
                                                      threshold=threshold)
            else:
                phase4d_unwrapped[..., i_t] = prelude(phase4d[..., i_t], mag4d[..., i_t], affine, mask=mask,
                                                      threshold=threshold)
        # Squeeze last dim if its shape is 1
        if phase.ndim == 2:
            phase_unwrapped = phase4d_unwrapped[..., 0, 0]
        elif phase.ndim == 3:
            phase_unwrapped = phase4d_unwrapped[..., 0]
        else:
            phase_unwrapped = phase4d_unwrapped

    else:
        raise RuntimeError(f'The unwrap function {unwrapper} is not implemented')

    return phase_unwrapped