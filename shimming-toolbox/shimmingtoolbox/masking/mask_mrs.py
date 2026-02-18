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

from shimmingtoolbox.utils import splitext, run_subprocess
from shimmingtoolbox.coils.coordinates import resample_from_to

logger = logging.getLogger(__name__)


def mask_mrs(fname_input, raw_datas, center, size):
    """
    Create a mask to shim single voxel MRS

    Args:
        fname_input (str): Input path of the fieldmap to be shimmed (supported extension .nii and .nii.gz)
        raw_datas (list): Input list of paths of the  raw-data (supported extension .rda, .spar/.sdat)
        center (list): Voxel's center position in mm of the x, y and z scanner coordinates
        size (list): Voxel size in mm of the x, y and z scanner coordinates
    Returns:
        numpy.ndarray: Cubic mask with the same dimensions as the MRS voxel.

    Notes:
        raw_datas is a list containing the required files for MRS conversion. Here are supported formats:
        - Single file .rda (Siemens)
        - Two files .spar/.sdat (Philips)
    """

    if fname_input is None:
        raise TypeError(f"The input field map needs to be specified")

    if raw_datas is None or len(raw_datas) == 0:
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
            # Handle rda files
            if is_rda_file(raw_datas[0]):
                if len(raw_datas) != 1:
                    raise ValueError("For .rda files, only one file should be provided")

                basename = os.path.basename(splitext(raw_datas[0])[0])
                run_subprocess(['spec2nii', 'rda', '-o', tmp, '-f', basename, raw_datas[0]])

            elif ((is_spar_file(raw_datas[0]) and is_sdat_file(raw_datas[1])) or
                  (is_spar_file(raw_datas[1]) and is_sdat_file(raw_datas[0]))):
                if len(raw_datas) != 2:
                    raise ValueError("For .spar/.sdat files, two files should be provided")

                # Order to sdat/spar
                if is_sdat_file(raw_datas[0]):
                    raw_data_sdat_spar = [raw_datas[0], raw_datas[1]]
                else:
                    raw_data_sdat_spar = [raw_datas[1], raw_datas[0]]

                basename = os.path.basename(splitext(raw_data_sdat_spar[0])[0])
                run_subprocess(['spec2nii', 'philips', '-o', tmp, '-f', basename, raw_data_sdat_spar[0], raw_data_sdat_spar[1]])
            else:
                raise ValueError("Unsupported raw_data format. Supported formats are: .rda, .spar/.sdat")

            fname_rawdata_nifti = os.path.join(tmp, basename + '.nii.gz')
            nii = nib.load(fname_rawdata_nifti)

            if nii.ndim == 4:
                data = np.ones_like(nii.get_fdata()[..., 0])
            else:
                data = np.ones_like(nii.get_fdata())

            nii_tmp = nib.Nifti1Image(data, nii.affine, nii.header)
            nii_target = nib.load(fname_input)
            return resample_from_to(nii_tmp, nii_target, mode='grid-constant', order=0, cval=0).get_fdata()

    # Handle no MRS data case (voxel size and center are given)
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
    mask[max(0, i - sd[0]):i + sd[0], max(0, j - sd[1]):j + sd[1], max(0, k - sd[2]):k + sd[2]] = 1

    return mask


def is_supported_file(file_path):
    return is_rda_file(file_path) or is_spar_file(file_path) or is_sdat_file(file_path)


def is_rda_file(file_path):
    _, ext = splitext(file_path)
    return ext.lower() == '.rda'


def is_spar_file(file_path):
    _, ext = splitext(file_path)
    return ext.lower() == '.spar'


def is_sdat_file(file_path):
    _, ext = splitext(file_path)
    return ext.lower() == '.sdat'
