#!usr/bin/env python3
# -*- coding: utf-8

import scipy.io
import numpy as np
import os


def load_vop(path_sar_file):
    if os.path.exists(path_sar_file):
        sar_data = scipy.io.loadmat(path_sar_file)
    else:
        raise FileNotFoundError('The SarDataUser.mat file could not be found.')
    # Assert file exists

    # Only return VOPs corresponding to 6 (body parts) and 8 (allowed forward power by channel)
    return sar_data['ZZ'][:, :, np.argwhere(np.logical_or(sar_data['ZZtype'] == 6, sar_data['ZZtype'] == 8))[:, 1]]
