#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Description

"""


import numpy as np
from shimmingtoolbox.unwrap.prelude import prelude


def unwrap_phase(complexArray, affine, unwrapFunction="prelude"):

    # TODO: create mask (here or in script?) (probably in script)
    # Call SCT or user defined mask
    mask = np.ones(complexArray.shape)
    if unwrapFunction == "prelude":
        unwrappedPhase = prelude(complexArray, affine, mask)
        return unwrappedPhase
    else:
        print("The unwrap function ", unwrapFunction, " is not implemented")
        return -1
