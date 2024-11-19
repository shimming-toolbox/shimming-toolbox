#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Creating MRS mask API
"""

import logging
import os.path

import nibabel as nib
import numpy as np
import numpy.linalg as npl
import pathlib
import tempfile

from shimmingtoolbox.utils import splitext
from shimmingtoolbox.utils import run_subprocess

logger = logging.getLogger(__name__)


def mask_mrs(fname_input, raw_data, center, size):
    """
    Create a mask to shim single voxel MRS

    Args:
        fname_input (str): Input path of the fieldmap to be shimmed (supported extension .nii and .nii.gz)
        raw_data (str): Input path of the siemens raw-data (supported extension .rda)
        center (list): Voxel's center position in mm of the x, y and z scanner coordinates
        size (list): Voxel size in mm of the x, y and z scanner coordinates
    Returns:
        numpy.ndarray: Cubic mask with the same dimensions as the MRS voxel.
    """

    if fname_input is None:
        raise TypeError(f"The input field map needs to be specified")

    if raw_data is None:
        logger.info("MRS raw data not provided, creating the mask with the given voxel position and size info")
        if center is None or size is None:
            raise TypeError('The raw_data is not provided; the voxel position and size are required to proceed')
        else:
            mrs_voxel_size = size
            logger.debug('center:', center)
            scanner_coordinate = np.array(center + (1, ))
            logger.debug(f"Scanner position is: {scanner_coordinate}")

    else:
        logger.info("Reading the raw_data")
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            basename = os.path.basename(raw_data)
            run_subprocess(['spec2nii', 'rda', '-o', tmp, '-f', basename, raw_data])
            fname_rawdata_nifti = os.path.join(tmp, basename + '.nii.gz')
            nii = nib.load(fname_rawdata_nifti)
            header_raw_data = nii.header
            affine = nii.affine
        position_sag = header_raw_data['qoffset_x']
        position_cor = header_raw_data['qoffset_y']
        position_tra = header_raw_data['qoffset_z']
        logger.debug(f"raw_data header: {header_raw_data}")
        logger.debug(f"affine: {affine}")
        scanner_coordinate = np.array([position_sag, position_cor, position_tra, 1])
        logger.debug(f"Scanner position is: {scanner_coordinate}")
        mrs_voxel_size = header_raw_data['pixdim'][1:4]

    logger.debug('mrs_voxel_size is:', mrs_voxel_size)
    fmap_nii = nib.load(fname_input)
    fmap_array = fmap_nii.get_fdata()
    fmap_affine = fmap_nii.affine
    fmap_header = fmap_nii.header
    logger.debug('reference_affine:', fmap_affine)
    logger.debug('reference_affine shape:', np.shape(fmap_affine))
    voxel_position = npl.inv(fmap_affine).dot(scanner_coordinate)
    voxel_position = np.round(voxel_position)
    logger.debug('voxel_position:', voxel_position)

    # The given coordinate (i, j, k) is the voxel's center position.
    i, j, k = map(int, voxel_position[:3])
    fmap_resolution = fmap_header['pixdim'][1:4]

    # The distance from the center of the MRS voxel to its edges is calculated based on number of fieldmap voxels.
    sd = np.ceil(mrs_voxel_size / (2 * fmap_resolution)).astype(int)

    # create a zero mask with the same size as the input fieldmap to be shimmed.
    mask = np.zeros(fmap_array.shape)

    # Change the MRS voxel position to have 1 value.
    mask[i - sd[0]:i + sd[0], j - sd[1]:j + sd[1], k - sd[2]:k + sd[2]] = 1
    return mask
