#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os
import sys
import subprocess

sys.path.insert(1, 'C:/Users/heuss/spinalcordtoolbox')
# from scripts.sct_create_mask import main as sct_mask

from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, help=f"Create a SCT (SpinalCordToolbox) mask.")
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS, help=f"Create a SCT (SpinalCordToolbox) mask from the input file. "
                                                      f"Return an output nifti file with SCT mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask.")
@click.option('-output', type=click.Path(), default=os.curdir, help="Name of output mask, Example: data.nii.")
@click.option('-process1', type=click.Choice(['coord', 'point', 'center', 'centerline']), default='center',
              help='Process to generate mask (coord, point, center or centerline)')
@click.option('-process2', default=None, help='Process to generate mask (XxY, file, None, file)')
@click.option('-size', default='41', help="Size of the mask in the axial plane, given in pixel (Example: 35) or in "
                                          "millimeter (Example: 35mm). If shape=gaussian, size corresponds to sigma ("
                                          "Example: 45).")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask: cylinder, box or gaussian.")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="Remove temporary files.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
def sct(fname_input, output, process1, process2, size, shape, remove, verbose):
    """
        Create a SCT (SpinalCordToolbox) mask from the input file. Return an output nifti file with SCT mask.

        Args:
            fname_input (file): Input nifti file to mask. Must be 3D. Example: data.nii.gz
            output (str): Name of output mask, Example: data.nii.
            process1 (str): Process to generate mask (coord, point, center or centerline).
                          <coord>: Center mask at the X,Y coordinates.
                          <point>: Center mask at the X,Y coordinates of
                          the label defined in input volume FILE.
                          <center>: Center mask in the middle of the FOV (nx/2,
                          ny/2).
                          <centerline>: At each slice, the mask is centered
                          at the spinal cord centerline, defined by the input
                          segmentation FILE. This segmentation file can be created with the CLI get_centerline.
                          (default: center)
            process2 (str): Process to generate mask.
                          For process1='coord': <XxY>: Center mask at the X,Y coordinates. (e.g.
                          "coord,20x15")
                          For process1='point': <FILE>: Center mask at the X,Y coordinates of
                          the label defined in input volume FILE. (e.g.
                          "point,label.nii.gz")
                          For process1='center': <(None)>: Center mask in the middle of the FOV (nx/2,
                          ny/2).
                          For process1='centerline': <FILE>: At each slice, the mask is centered
                          at the spinal cord centerline, defined by the input
                          segmentation FILE. This segmentation file can be created with the CLI get_centerline. 
                          (e.g. "centerline,t2_seg.nii.gz") (default: center)
            size: Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example:
                    35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41).
            shape (str): Shape of the mask (default: cylinder).
            remove (int): Remove temporary files (default: 1).
            verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended (default: 1).

        Returns:
            output (str): Output nifti file with SCT mask.
        """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(output):
        os.makedirs(output)
    
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
