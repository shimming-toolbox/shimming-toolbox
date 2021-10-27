#!usr/bin/env python3
# -*- coding: utf-8

import scipy.io
import os


def load_siemens_vop(path_sar_file):
    """

    Args:
        path_sar_file: Path to the 'SarDataUser.mat' file containing the scanner's VOPs. This file should be available
        at the scanner in 'C:/Medcom/MriProduct/PhysConfig'

    Returns:
        numpy.ndarray: VOP matrices (n_coils, n_coils, n_VOPs)

    """
    if os.path.exists(path_sar_file):
        sar_data = scipy.io.loadmat(path_sar_file)
    else:
        raise FileNotFoundError('The SarDataUser.mat file could not be found.')
    # Assert file exists

    if 'ZZ' not in sar_data:
        raise ValueError('The SAR data does not contain the expected VOP values.')

    return sar_data['ZZ']
    # Only return VOPs corresponding to 6 (body parts) and 8 (allowed forward power by channel)
    # return sar_data['ZZ'][:, :, np.argwhere(np.logical_or(sar_data['ZZtype'] == 6, sar_data['ZZtype'] == 8))[:, 1]]
