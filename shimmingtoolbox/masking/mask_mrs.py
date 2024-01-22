#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Creating MRS mask API
"""
import os
import numpy as np
import twixtools
import numpy.linalg as npl
import nibabel as nib
import logging
import json
import spec2nii
import subprocess


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mask_mrs(fname_input, raw_data, X, Y, Z, V):
    """
    Create a mask to shim single voxel MRS

    Args:
        fname_input: Input path of the fieldmap to be shimmed (both nii.gz and json file should be in this path)
        raw_data: Input path of the of the twix raw-data (supported extention .dat) [optional]
        X: scanner's X position in mm [optional]
        Y: scanner's Y position in mm [optional]
        Z: scanner's Z position in mm [optional]
        V: MRS voxel size in mm [optional]

    Returns:
        numpy.ndarray: Cubic mask with same dimensions as MRS voxel.
    """

    if fname_input is None:
        raise FileNotFoundError(f"The file '{fname_input}' was not found. See: st_mask mask_mrs -h")

    if X is not None and Y is not None and Z is not None and V is not None:
        # If X,Y,Z,V arguments were provided, execute this block
        MRS_voxel_thick = V
        scanner_coordinate = np.array([X, Y, Z, 1])
        logger.info(f"Scanner position: {scanner_coordinate}")

    else:
        # If X,Y,Z,V arguments were NOT provided, execute this block
        if raw_data is None:
            raise FileNotFoundError(f"The file '{raw_data}' was not found. If X,Y,Z and V are not directly given the raw-data must be given to read these info, See: st_mask mask_mrs -h")
        else:
            logger.info("Reading the twix raw-data")
            subprocess.run(f"spec2nii twix {raw_data} -e image", shell=True )
            name_nii = os.path.splitext(raw_data)[0]
            header_twix= nib.load(name_nii + '.nii.gz').header
            affine= nib.load(name_nii + '.nii.gz').affine
            position_sag= header_twix['qoffset_x']
            position_cor = header_twix['qoffset_y']
            position_tra = header_twix['qoffset_z']
            logger.info(f"twix header: {header_twix}")
            logger.info(f"affine: {affine}")
            scanner_coordinate= np.array([position_sag, position_cor, position_tra, 1])
            logger.info(f"Scanner position: {scanner_coordinate}")


    MRS_voxel_thick = header_twix['pixdim'][1]
    logger.info('MRS_voxel_thick is:', MRS_voxel_thick)
    fmap_nii= nib.load(fname_input)
    fmap_array= fmap_nii.get_fdata()
    fmap_affine= fmap_nii.affine
    logger.info('reference_affine:', fmap_affine)
    logger.info('reference_affine shape:', np.shape(fmap_affine))
    voxel_position= npl.inv(fmap_affine).dot(scanner_coordinate)
    voxel_position = np.round(voxel_position)
    logger.info('voxel_position:', voxel_position)
    I, J, K = map(int, voxel_position[:3])
    data_dir = os.path.dirname(fname_input)

    # Get the file name of the input without extension
    json_name = os.path.splitext(os.path.basename(fname_input))[0][:-4]

    # Combine the directory and file name without extension
    json_dir = os.path.join(data_dir, json_name)

    # Open the JSON file
    json_file = json_dir+'.json'

    with open(json_file, 'r') as info:
        # Load the data from the json file
        header_dcm = json.load(info)

    # Read the slice thickness of the reference GRE image.
    slice_thickness = header_dcm['SliceThickness'] # Isotropic fieldmap (mm)

    # The given XYZ is the voxel's center position.
    # To create a mask the distance from center of MRS voxel to its edge should be calculated based on number of fieldmap voxels.
    SD = int(np.ceil(MRS_voxel_thick/(2*slice_thickness)))

    # create a zero mask with the same size as the input fieldmap to be shimmed.
    mask = np.zeros(fmap_array.shape)

    # Change the MRS voxel position to have 1 value.
    mask[I - SD:I + SD, J - SD:J + SD, K - SD:K + SD] = 1
    return mask
