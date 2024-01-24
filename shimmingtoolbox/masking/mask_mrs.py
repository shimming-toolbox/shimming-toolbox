#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Creating MRS mask API
"""
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import logging
from shimmingtoolbox.utils import splitext
from shimmingtoolbox.utils import run_subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mask_mrs(fname_input, raw_data, center, size):
    """
    Create a mask to shim single voxel MRS

    Args:
        fname_input (str): Input path of the fieldmap to be shimmed (supported extention .nii and .nii.gz) [required]
        raw_data (str): Input path of the of the twix raw-data (supported extention .dat) [optional]
        center (float): voxel's center position in mm of the x, y and z of the scanner's coordinate [optional]
        size (float): voxel size in mm of the x, y and z of the scanner's coordinate [optional]

    Returns:
        numpy.ndarray: Cubic mask with same dimensions as MRS voxel.
    """

    if fname_input is None:
        raise TypeError(f"The file '{fname_input}' is required. See: st_mask mrs-mask -h")

    if raw_data is None:
        logger.info(" The raw_data is not provided, creating the mask with the given voxel position and size info")
        if center is None or size is None:
             raise TypeError('As the raw_data is not provided, the voxel position and size must be given to create the mask (See: st_mask mrs-mask -h)')
        else:
            mrs_voxel_size = size
            logger.debug('center:',center )
            scanner_coordinate = np.array(center + (1,))
            logger.debug(f"Scanner position is: {scanner_coordinate}")

    else:
            logger.info("Reading the twix raw_data")
            run_subprocess(['spec2nii', 'twix', raw_data, '-e', 'image'])
            name_nii, ext = splitext(raw_data)
            header_twix = nib.load(name_nii + '.nii.gz').header
            affine = nib.load(name_nii + '.nii.gz').affine
            position_sag = header_twix['qoffset_x']
            position_cor = header_twix['qoffset_y']
            position_tra = header_twix['qoffset_z']
            logger.debug(f"twix header: {header_twix}")
            logger.debug(f"affine: {affine}")
            scanner_coordinate = np.array([position_sag, position_cor, position_tra, 1])
            logger.debug(f"Scanner position is: {scanner_coordinate}")
            mrs_voxel_size = header_twix['pixdim'][1:4]


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

      # The given coordinate (I, J, K) is the voxel's center position.
    I, J, K = map(int, voxel_position[:3])
    fmap_resolution = fmap_header['pixdim'][1:4]

    # To create a mask, the distance from the center of the MRS voxel to it's edges should be calculated based on number of fieldmap voxels.
    half_voxel_distances = np.ceil(np.divide(mrs_voxel_size, 2 * fmap_resolution)).astype(int)
    sd = half_voxel_distances

    # create a zero mask with the same size as the input fieldmap to be shimmed.
    mask = np.zeros(fmap_array.shape)

    # Change the MRS voxel position to have 1 value.
    mask[I - sd[0]:I + sd[0], J - sd[1]:J + sd[1], K - sd[2]:K + sd[2]] = 1
    return mask
