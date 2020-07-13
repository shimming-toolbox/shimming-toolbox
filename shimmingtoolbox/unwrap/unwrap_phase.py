#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Description

"""


import numpy as np
from shimmingtoolbox.unwrap.prelude import prelude


def unwrap_phase(complex_array, affine, unwrap_function="prelude"):

    # TODO: create mask (here or in script?) (probably in script)
    # Call SCT or user defined mask
    # mask = np.ones(complex_array.shape)

    unwrapped_phase = np.empty(complex_array.shape, np.complex)
    if unwrap_function == "prelude":

        if complex_array.ndim == 2:
            # TODO: implement
            raise Exception('Not implemented')
        elif complex_array.ndim == 3:
            unwrapped_phase = prelude(complex_array, affine)
        elif complex_array.ndim == 4:
            for i_Echo in range(complex_array.shape[3]):
                unwrapped_phase[:, :, :, i_Echo] = prelude(complex_array[:, :, :, i_Echo], affine)
        elif complex_array.ndim == 5:
            for i_Echo in range(complex_array.shape[3]):
                for i_acq in range(complex_array.shape[4]):
                    unwrapped_phase[:, :, :, i_Echo, i_acq] = prelude(complex_array[:, :, :, i_Echo, i_acq], affine)
        else:
            # TODO: Better error handling
            raise Exception('Number of dimensions not supported')

    else:
        raise Exception('The unwrap function ', unwrap_function, ' is not implemented')

    return unwrapped_phase
