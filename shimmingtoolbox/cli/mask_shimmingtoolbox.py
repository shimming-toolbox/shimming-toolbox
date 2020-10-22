#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def main():
    pass


@main.command(context_settings=CONTEXT_SETTINGS,
              help=f"Apply a cube mask to the input file. Returns data with cube mask.")
@click.option("-fname_data", required=True, help="Full file path and name of the data to mask")
# @click.option("-fname_data", required=True, type=click.Path(exists=True, file_okay=False), help="Input path of nifti "
#  "file to mask")
@click.option("-len_dim1", required=True, type=int, help="Length of the side of the cube along first dimension (in "
                                                         "pixels)")
@click.option("-len_dim2", required=True, type=int, help="Length of the side of the cube along second dimension (in "
                                                         "pixels)")
@click.option("-len_dim3", required=True, type=int, help="Length of the side of the cube along third dimension (in "
                                                         "pixels)")
@click.option("-center_dim1", type=int, default=None, help="Center of the cube along first dimension (in "
                                                           "pixels). If no center is provided, the middle is "
                                                           "used.")
@click.option("-center_dim2", type=int, default=None, help="Center of the cube along second dimension (in "
                                                           "pixels). If no center is provided, the middle is "
                                                           "used.")
@click.option("-center_dim3", type=int, default=None, help="Center of the cube along third dimension (in "
                                                           "pixels). If no center is provided, the middle is "
                                                           "used.")
def cube(fname_data, len_dim1, len_dim2, len_dim3, center_dim1, center_dim2, center_dim3):
    """
        Apply a cube mask to the input file. Returns data with cube mask.

        Args:
            fname_data (str): Path of data to mask.
            len_dim1 (int): Length of the side of the cube along first dimension (in pixels).
            len_dim2 (int): Length of the side of the cube along second dimension (in pixels).
            len_dim3 (int): Length of the side of the cube along third dimension (in pixels).
            center_dim1 (int): Center of the cube along first dimension (in pixels). If no center is
                            provided, the middle is used.
            center_dim2 (int): Center of the cube along second dimension (in pixels). If no center is
                            provided, the middle is used.
            center_dim3 (int): Center of the cube along third dimension (in pixels). If no center is
                            provided, the middle is used.

        Return:
            (numpy.ndarray): Data with cube mask applied.
        """
    path = os.path.join(fname_data)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 3:
        mask_cb = shape_cube(data, len_dim1, len_dim2, len_dim3, center_dim1, center_dim2, center_dim3)  # creation
        # of the cube mask
        return data * mask_cb  # application of the mask on the data

    else:
        return None


@main.command(context_settings=CONTEXT_SETTINGS,
              help=f"Apply a square mask to the input file. Returns data with square mask.")
@click.option("-fname_data", required=True, help="Full file path and name of the data to mask")
@click.option("-len_dim1", required=True, type=int, help="Length of the side of the square along first dimension (in "
                                                         "pixels)")
@click.option("-len_dim2", required=True, type=int, help="Length of the side of the square along second dimension (in "
                                                         "pixels)")
@click.option("-center_dim1", type=int, default=None, help="Center of the square along first dimension (in "
                                                           "pixels). If no center is provided, the middle is "
                                                           "used.")
@click.option("-center_dim2", type=int, default=None, help="Center of the square along second dimension (in "
                                                           "pixels). If no center is provided, the middle is "
                                                           "used.")
def square(fname_data, len_dim1, len_dim2, center_dim1, center_dim2):
    """
            Apply a square mask to the input file. Returns data with square mask.

            Args:
                fname_data (str): Data to mask.
                len_dim1 (int): Length of the side of the square along first dimension (in pixels).
                len_dim2 (int): Length of the side of the square along second dimension (in pixels).
                center_dim1 (int): Center of the square along first dimension (in pixels). If no center is
                                provided, the middle is used.
                center_dim2 (int): Center of the square along second dimension (in pixels). If no center is
                                provided, the middle is used.

            Return:
                (numpy.ndarray): Data with cube mask applied.
            """
    path = os.path.join(fname_data)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 2:
        mask_sqr = shape_square(data, len_dim1, len_dim2, center_dim1, center_dim2)  # creation of the mask
        return data * mask_sqr  # application of the mask on the data

    elif len(data.shape) == 3:
        for z in range(data.shape[2]):
            mask_sqr = np.zeros_like(data)  # initialization of 3D array of zeros
            img_2d = data[:, :, z]  # extraction of a MRI slice (2D)
            mask = shape_square(img_2d, len_dim1, len_dim2, center_dim1, center_dim2)  # creation of the mask on each
            # slice (2D)
            mask_sqr[:, :, z] = mask  # addition of each masked slice to form a 3D array
            return data * mask_sqr  # application of the mask on the data

    else:
        return None


@main.command(context_settings=CONTEXT_SETTINGS,
              help=f"Apply a threshold mask to the input file. Returns data with threshold mask.")
@click.option("-fname_data", required=True, help="Full file path and name of the data to mask")
@click.option("-thr", default=30, help="Value to threshold the data: voxels will be set to zero if their value is "
                                       "equal or less than this threshold")
def mask_threshold(fname_data, thr):
    """
        Apply a threshold mask to the input file. Returns data with threshold mask.

        Args:
            fname_data (str): Data to be masked
            thr: Value to threshold the data: voxels will be set to zero if their
                value is equal or less than this threshold

        Returns:
            (numpy.ndarray): Data with threshold mask applied.
        """
    path = os.path.join(fname_data)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = threshold(data, thr)  # creation of the mask

    return data * mask_thr  # application of the mask on the data


if __name__ == '__main__':
    main()
