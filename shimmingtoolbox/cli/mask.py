#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

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

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, output)
    click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
    return output


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Creates a SCT (SpinalCordToolbox) mask from the input file. Depending on the shape (cylinder,"
                       f" box or Gaussian), a mask is created along z direction. To generate this mask, its center"
                       f" must be specified by the user according to 4 processes."
                       f"Return an output nifti file with SCT mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input nifti file to mask. Must be 3D. Supported extensions are .nii or .nii.gz. Example: "
                   "data.nii.gz")
@click.option('-output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. Example: data.nii. (default:"
                   " (os.curdir, 'mask.nii.gz')).")
@click.option('-process1', type=click.Choice(['coord', 'point', 'center', 'centerline']), default='center',
              help="(str): First process to generate mask: "
                   "<coord>: Center mask at the X,Y coordinates. "
                   "<point>: Center mask at the X,Y coordinates of the label defined in input volume file"
                   "<center>: Center mask in the middle of the FOV (nx/2, ny/2). "
                   "<centerline>: At each slice, the mask is centered at the spinal cord centerline, defined by the "
                   "input segmentation FILE. This segmentation file can be created with the CLI get_centerline. "
                   "(default: center)")
@click.option('-process2', default=None,
              help="(str): Second process to generate mask: "
                   "For process1='coord': <XxY>: Center mask at the X,Y coordinates. (e.g. 'coord,20x15')."
                   "For process1='point': <FILE>: Center mask at the X,Y coordinates of the label defined in input "
                   "volume FILE. (e.g. 'point,label.nii.gz')."
                   "For process1='center': <(None)>: Center mask in the middle of the FOV (nx/2, ny/2)."
                   "For process1='centerline': <FILE>: At each slice, the mask is centered at the spinal cord "
                   "centerline, defined by the input segmentation FILE. This segmentation file can be created with the"
                   " CLI get_centerline. (e.g. 'centerline,t2_seg.nii.gz'). (default: None)")
@click.option('-size', default='41',
              help="(str): Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter "
                   "(Example: 35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41)")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="(str): Shape of the mask. (default: cylinder)")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="(int): Remove temporary files. (default: 1)")
@click.option('-verbose', type=click.IntRange(0, 2), default=1,
              help="(int): Verbose: 0 = nothing, 1 = classic, 2 = expended. (default: 1)")
def sct(fname_input, output, process1, process2, size, shape, remove, verbose):
    if process1 == "center" and process2 is None:
        subprocess.run(['sct_create_mask', '-i', fname_input, '-p', process1, '-size', size, '-f', shape, '-o', output,
                        '-r', str(remove), '-v', str(verbose)], check=True)

    elif process1 == "center" and process2 is not None:
        raise ValueError("The process 'center' must not have a 2nd argument in process2.")

    else:
        subprocess.run(['sct_create_mask', '-i', fname_input, '-p', process1 + "," + process2, '-size', size, '-f',
                        shape, '-o', output, '-r', str(remove), '-v', str(verbose)], check=True)

    click.echo(f"The path for the output mask is: {os.path.abspath(output)}")
    return output
