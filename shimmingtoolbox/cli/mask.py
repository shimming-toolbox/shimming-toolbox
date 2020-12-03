#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os
import sys
import subprocess

from shimmingtoolbox.masking.threshold import threshold
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
                        '-r', remove, '-v', verbose], check=True)

    elif process1 == "center" and process2 is not None:
        raise ValueError("The process 'center' must not have a 2nd argument in process2.")

    else:
        subprocess.run(
            ['sct_create_mask', '-i', fname_input, '-p', process1 + "," + process2, '-size', size, '-f', shape,
             '-o', output, '-r', remove, '-v', verbose], check=True)

    click.echo(f"The path for the output mask is: {os.path.abspath(output)}")
    return output
