#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, help=f"Create a mask based on a specified shape (cube, square, etc.) "
                                                     f"or based on the thresholding of an input image.")
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
              help=f"Create a cube mask from the input file. Return the filename for the output mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output folder for mask in nifti file")
@click.option("-size", required=True, type=int, help="Length of the side of the cube along first, second and third "
                                                     "dimension (in pixels)")
@click.option("-center", nargs=3, type=int, default=(None, None, None), help="Center of the cube along first, second "
                                                                             "and third dimension (in pixels). If no "
                                                                             "center is provided, the middle is used.")
def cube(fname_input, output, size, center):
    """
        Create a cube mask from the input file. Return an output nifti file with cube mask.

        Args:
            fname_input (str): Complete input path of the nifti file to mask.
            output (str): Output folder for cube mask.
            size (int): Length of the side of the cube along first, second and third dimension (in pixels).
            center (int): Center of the cube along first, second and third dimension (in pixels). If no center is
                            provided, the middle is used.
            
        Return:
            output (str): Filename for the output mask.
        """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(output):
        os.makedirs(output)

    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 3:
        mask_cb = shape_cube(data, size, size, size, center[0], center[1], center[2])  # creation
        # of the cube mask
        mask_cb = mask_cb.astype(int)
        nii_img = nib.Nifti1Image(mask_cb, nii.affine)
        fname_mask = os.path.join(output, 'mask.nii.gz')
        nib.save(nii_img, fname_mask)
        click.echo(f"The filename for the output mask is: {os.path.abspath(fname_mask)}")
        return output

    else:
        raise ValueError("The nifti file does not have 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
              help=f"Create a square mask from the input file. Return an output nifti file with square mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output folder for mask in nifti file")
@click.option("-size", required=True, type=int, help="Length of the side of the square along first and second dimension"
                                                     " (in pixels)")
@click.option("-center", nargs=2, type=int, default=(None, None), help="Center of the cube along first and second "
                                                                       "dimension (in pixels). If no center is "
                                                                       "provided, the middle is used.")
def square(fname_input, output, size, center):
    """
            Create a square mask from the input file. Return an output nifti file with square mask.

            Args:
                fname_input (str): Complete input path of the nifti file to mask.
                output (str): Output folder for square mask.
                size (int): Length of the side of the square along first and second dimension (in pixels).
                center (int): Center of the cube along first and second dimension (in pixels). If no center is
                            provided, the middle is used.
                
            Return:
                output (str): Filename for the output mask.
            """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(output):
        os.makedirs(output)

    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array
    fname_mask = os.path.join(output, 'mask.nii.gz')

    if len(data.shape) == 2:
        mask_sqr = shape_square(data, size, size, center[0], center[1])  # creation of the square mask
        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, fname_mask)
        click.echo(f"The filename for the output mask is: {os.path.abspath(fname_mask)}")
        return output

    elif len(data.shape) == 3:
        for z in range(data.shape[2]):
            mask_sqr = np.zeros_like(data)  # initialization of 3D array of zeros
            img_2d = data[:, :, z]  # extraction of a MRI slice (2D)
            mask_slice = shape_square(img_2d, size, size, center[0], center[1])  # creation of the mask
            # on each slice (2D)
            mask_sqr[:, :, z] = mask_slice  # addition of each masked slice to form a 3D array
            mask_sqr = mask_sqr.astype(int)
            nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
            nib.save(nii_img, fname_mask)
            click.echo(f"The filename for the output mask is: {os.path.abspath(fname_mask)}")
            return output

    else:
        raise ValueError("The nifti file does not have 2 or 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
              help=f"Create a threshold mask from the input file. Return an output nifti file with threshold mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file to mask")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output folder for mask in nifti file")
@click.option("-thr", default=30, help="Value to threshold the data: voxels will be set to zero if their value is "
                                       "equal or less than this threshold")
def threshold(fname_input, output, thr):
    """
        Create a threshold mask from the input file. Return an output nifti file with threshold mask.

        Args:
            fname_input (str): Complete input path of the nifti file to mask.
            output (str): Output folder for square mask.
            thr: Value to threshold the data: voxels will be set to zero if their
                value is equal or less than this threshold

        Returns:
            output (str): Filename for the output mask.
        """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(output):
        os.makedirs(output)

    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    fname_mask = os.path.join(output, 'mask.nii.gz')
    nib.save(nii_img, fname_mask)
    click.echo(f"The filename for the output mask is: {os.path.abspath(fname_mask)}")
    return output
