#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube
from shimmingtoolbox.utils import create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS,
             help=f"Create a mask based on a specified shape (box, rectangle, SpinalCord Toolbox mask) or based on the "
                  f"thresholding of an input image. Callable with the prefix 'st' in front of 'mask'. "
                  f"(Example: 'st_mask -h').")
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a box mask from the input file. The nifti file is converted to a numpy array. If this "
                       f"array is in 3D dimensions, then a binary mask is created from this array in the form of a box"
                       f" with lengths defined in 'size'. This box is centered according to the 3 dimensions indicated"
                       f" in 'center'. The mask is stored by default under the name 'mask.nii.gz' in the output folder."
                       f"Return the filename for the output mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. This nifti file must have 3D. Supported extensions are"
                   " .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-size', nargs=3, required=True, type=int,
              help="(int): Length of the side of the box along first, second and third dimension (in pixels). "
                   "(nargs=3)")
@click.option('-center', nargs=3, type=int, default=(None, None, None),
              help="(int): Center of the box along first, second and third dimension (in pixels). If no center "
                   "is provided (None), the middle is used. (nargs=3) (default: None, None, None)")
def box(fname_input, output, size, center):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array
    create_output_dir(output, is_file=True)

    if len(data.shape) == 3:
        mask_cb = shape_cube(data, size[0], size[1], size[2], center[0], center[1], center[2])  # creation
        # of the box mask
        mask_cb = mask_cb.astype(int)
        nii_img = nib.Nifti1Image(mask_cb, nii.affine)
        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    else:
        raise ValueError("The nifti file does not have 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a rectangle mask from the input file. "
                       f"The nifti file is converted to a numpy array. If this array is in 2 dimensions, then a binary"
                       f" mask is created from this array in the form of a rectangle of lengths defined in 'size'. This"
                       f" rectangle is centered according to the 2 dimensions indicated in 'center'. If this array is "
                       f"in 3 dimensions, a binary mask is created in the shape of rectangle for each slice of the 3rd"
                       f" dimension of the array, in the same way as for a 2D array. The masks of all these slices are "
                       f"grouped in an array to form a binary mask in 3 dimensions. The mask is stored by default under"
                       f" the name 'mask.nii.gz' in the output folder."
                       f"Return an output nifti file with square mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. This nifti file must have 2D or 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.curdir,
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-size', nargs=2, required=True, type=int,
              help="(int): Length of the side of the box along first and second dimension (in pixels). (nargs=2)")
@click.option('-center', nargs=2, type=int, default=(None, None),
              help="(int): Center of the box along first and second dimension (in pixels). If no center is "
                   "provided (None), the middle is used. (nargs=2) (default: None, None)")
def rect(fname_input, output, size, center):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array
    create_output_dir(output, is_file=True)

    if len(data.shape) == 2:
        mask_sqr = shape_square(data, size[0], size[1], center[0], center[1])  # creation of the rectangle mask
        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)

        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    elif len(data.shape) == 3:
        mask_sqr = np.zeros_like(data)  # initialization of 3D array of zeros
        for z in range(data.shape[2]):
            img_2d = data[:, :, z]  # extraction of a MRI slice (2D)
            mask_slice = shape_square(img_2d, size[0], size[1], center[0], center[1])  # creation of the mask
            # on each slice (2D)
            mask_sqr[:, :, z] = mask_slice  # addition of each masked slice to form a 3D array

        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    else:
        raise ValueError("The nifti file does not have 2 or 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a threshold mask from the input file. "
                       f"The nifti file is converted into a numpy array. A binary mask is created from the thresholding"
                       f" of the array. The mask is stored by default under the name 'mask.nii.gz' in the output "
                       f"folder. Return an output nifti file with threshold mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. Supported extensions are .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.curdir,
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-thr', default=30, help="(int): Value to threshold the data: voxels will be set to zero if their "
                                       "value is equal or less than this threshold. (default: 30)")
def threshold(fname_input, output, thr):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array
    create_output_dir(output, is_file=True)

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, output)
    click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
    return output
