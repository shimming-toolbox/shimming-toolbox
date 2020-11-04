#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, help=f"Creates a cube, square or threshold mask.")
def mask():
    pass


@mask.command(context_settings=CONTEXT_SETTINGS,
              help=f"Creates a cube mask from the input file. Returns an output nifti file with cube mask.")
@click.option('-input', type=click.Path(), required=True, help="Complete input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output path for mask in nifti file")
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
def cube(input, output, len_dim1, len_dim2, len_dim3, center_dim1, center_dim2, center_dim3):
    """
        Creates a cube mask from the input file. Returns an output nifti file with cube mask.

        Args:
            input (str): Complete input path of the nifti file to mask.
            output (str): Output nifti file for cube mask.
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
            output (str): Output nifti file with cube mask.
        """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(input):
        raise FileNotFoundError("No nifti path found")
    if not os.path.exists(output):
        os.makedirs(output)

    path = os.path.join(input)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 3:
        mask_cb = shape_cube(data, len_dim1, len_dim2, len_dim3, center_dim1, center_dim2, center_dim3)  # creation
        # of the cube mask
        mask_cb = mask_cb.astype(int)
        nii_img = nib.Nifti1Image(mask_cb, nii.affine)
        nib.save(nii_img, os.path.join(output, 'mask.nii.gz'))
        click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
                   % os.path.abspath(output))
        return output

    else:
        raise ValueError("The nifti file does not have 3 dimensions.")


@mask.command(context_settings=CONTEXT_SETTINGS,
              help=f"Creates a square mask from the input file. Returns an output nifti file with square mask.")
@click.option('-input', type=click.Path(), required=True, help="Complete input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output path for mask in nifti file")
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
def square(input, output, len_dim1, len_dim2, center_dim1, center_dim2):
    """
            Creates a square mask from the input file. Returns an output nifti file with square mask.

            Args:
                input (str): Complete input path of the nifti file to mask.
                output (str): Output nifti file for square mask.
                len_dim1 (int): Length of the side of the square along first dimension (in pixels).
                len_dim2 (int): Length of the side of the square along second dimension (in pixels).
                center_dim1 (int): Center of the square along first dimension (in pixels). If no center is
                                provided, the middle is used.
                center_dim2 (int): Center of the square along second dimension (in pixels). If no center is
                                provided, the middle is used.

            Return:
                output (str): Output nifti file with square mask.
            """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(input):
        raise FileNotFoundError("No nifti path found")
    if not os.path.exists(output):
        os.makedirs(output)

    path = os.path.join(input)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 2:
        mask_sqr = shape_square(data, len_dim1, len_dim2, center_dim1, center_dim2)  # creation of the square mask
        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, os.path.join(output, 'mask.nii.gz'))
        click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
                   % os.path.abspath(output))
        return output

    elif len(data.shape) == 3:
        for z in range(data.shape[2]):
            mask_sqr = np.zeros_like(data)  # initialization of 3D array of zeros
            img_2d = data[:, :, z]  # extraction of a MRI slice (2D)
            mask_slice = shape_square(img_2d, len_dim1, len_dim2, center_dim1, center_dim2)  # creation of the mask
            # on each slice (2D)
            mask_sqr[:, :, z] = mask_slice  # addition of each masked slice to form a 3D array
            mask_sqr = mask_sqr.astype(int)
            nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
            nib.save(nii_img, os.path.join(output, 'mask.nii.gz'))
            click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
                       % os.path.abspath(output))
            return output

    else:
        raise ValueError("The nifti file does not have 2 or 3 dimensions.")


@mask.command(context_settings=CONTEXT_SETTINGS,
              help=f"Creates a threshold mask from the input file. Returns an output nifti file with threshold mask.")
@click.option('-input', type=click.Path(), required=True, help="Complete input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output path for mask in nifti file")
@click.option("-thr", default=30, help="Value to threshold the data: voxels will be set to zero if their value is "
                                       "equal or less than this threshold")
def mask_threshold(input, output, thr):
    """
        Creates a threshold mask from the input file. Returns an output nifti file with threshold mask.

        Args:
            input (str): Complete input path of the nifti file to mask.
            output (str): Output nifti file for square mask.
            thr: Value to threshold the data: voxels will be set to zero if their
                value is equal or less than this threshold

        Returns:
            output (str): Output nifti file with threshold mask.
        """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(input):
        raise FileNotFoundError("No nifti path found")
    if not os.path.exists(output):
        os.makedirs(output)

    path = os.path.join(input)
    nii = nib.load(path)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, os.path.join(output, 'mask.nii.gz'))
    click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
               % os.path.abspath(output))
    return output
