#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Wrapper to different unwrapping algorithms

"""


import numpy as np
from shimmingtoolbox.unwrap.prelude import prelude


def unwrap_phase(complex_array, affine, unwrap_function="prelude", mask=np.array([-1])):
    """ Calls different unwrapping algorithms according to the specified `unwrap_function` parameter

    Args:
        complex_array (numpy.ndarray): 2D,3D,4D or 5D complex values numpy array to perform phase unwrapping
        affine (numpy.ndarray): 2D array (4x4) containing the transformation coefficients. Can be acquired by :
            nii = nib.load("nii_path")
            affine = nii.affine
        unwrap_function (string, optional): unwrapper algorithm name. Possible values : prelude
        mask: numpy array of booleans with shape of `complex_array` to mask during phase unwrapping. Calling `mask` wth
        np.array([-1]) is the same as specifying no mask.

    Returns:
        numpy.ndarray: 3D array with the shape of `complex_array` of the unwrapped phase output from prelude

    Raises:
        Exception: complex array does not have 2,3,4,5 dimensions
        Exception: unwrapping algorithm in `unwrap_function` is not supported
    """
    unwrapped_phase = np.empty(complex_array.shape, np.complex)
    if unwrap_function == "prelude":

        if complex_array.ndim == 2:
            complex_array = complex_array[..., np.newaxis]
            unwrapped_phase = prelude(complex_array, affine, mask)
        elif complex_array.ndim == 3:
            unwrapped_phase = prelude(complex_array, affine, mask)
        elif complex_array.ndim == 4:
            for i_Echo in range(complex_array.shape[3]):
                unwrapped_phase[:, :, :, i_Echo] = prelude(complex_array[:, :, :, i_Echo], affine, mask)
        elif complex_array.ndim == 5:
            for i_Echo in range(complex_array.shape[3]):
                for i_acq in range(complex_array.shape[4]):
                    unwrapped_phase[:, :, :, i_Echo, i_acq] = prelude(complex_array[:, :, :, i_Echo, i_acq], affine,
                                                                      mask)
        else:
            raise Exception('Number of dimensions not supported')

    else:
        raise Exception('The unwrap function ', unwrap_function, ' is not implemented')

    return unwrapped_phase
