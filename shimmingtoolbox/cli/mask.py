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


@click.group(context_settings=CONTEXT_SETTINGS, help=f"Creates a cube, square or threshold mask.")
def mask():
    pass


@mask.command(context_settings=CONTEXT_SETTINGS, help=f"Creates a SCT (SpinalCordToolbox) mask from the input file "
                                                      f"with coordinates process. Returns an output nifti file with "
                                                      f"SCT mask.")
@click.option('-input', type=click.File('r'), required=True, help="Complete input path of the nifti file to mask.")
@click.option('-output', type=click.Path(), default=os.curdir, help="Name of output mask, Example: data.nii.")
@click.option('-size', default='41', help="Size of the mask in the axial plane, given in pixel (Example: 35) or in "
                                          "millimeter (Example: 35mm). If shape=gaussian, size corresponds to sigma ("
                                          "Example: 45).")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask: cylinder, box or gaussian.")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="Remove temporary files.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
@click.option('-process', required=True, help='<coord,XxY>: Center mask at the X,Y coordinates. (e.g. "coord,20x15")')
def sct_coord(input, output, size, shape, remove, verbose, process):
    """
        Creates a SCT (SpinalCordToolbox) mask from the input file with coordinates process. Returns an output nifti
        file with SCT mask.

        Args:
            input (file): Input nifti file to mask. Must be 3D. Example: data.nii.gz
            output (str): Name of output mask, Example: data.nii.
            size: Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example:
                    35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41).
            shape (str): Shape of the mask (default: cylinder).
            remove (int): Remove temporary files (default: 1).
            verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended (default: 1).
            process (str): Process to generate mask.
                          <coord,XxY>: Center mask at the X,Y coordinates. (e.g.
                          "coord,20x15")

        Returns:
            output (str): Output nifti file with SCT mask.
        """
    subprocess.run(['sct_create_mask', '-i', input, '-p', 'coord', process, '-size', size, '-f', shape, 'o', output,
                    '-r', remove, '-v', verbose], check=True)
    click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
               % os.path.abspath(output))
    return output


@mask.command(context_settings=CONTEXT_SETTINGS, help=f"Creates a SCT (SpinalCordToolbox) mask from the input file "
                                                      f"with point process. Returns an output nifti file with "
                                                      f"SCT mask.")
@click.option('-input', type=click.File('r'), required=True, help="Complete input path of the nifti file to mask.")
@click.option('-output', type=click.Path(), default=os.curdir, help="Name of output mask, Example: data.nii.")
@click.option('-size', default='41', help="Size of the mask in the axial plane, given in pixel (Example: 35) or in "
                                          "millimeter (Example: 35mm). If shape=gaussian, size corresponds to sigma ("
                                          "Example: 45).")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask: cylinder, box or gaussian.")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="Remove temporary files.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
@click.option('-process', type=click.File('r'), required=True,
              help='<point,FILE>: Center mask at the X,Y coordinates of the label defined in input volume FILE. ('
                   'e.g."point,label.nii.gz")')
def sct_point(input, output, size, shape, remove, verbose, process):
    """
        Creates a SCT (SpinalCordToolbox) mask from the input file with point process. Returns an output nifti file with
         SCT mask.

        Args:
            input (file): Input nifti file to mask. Must be 3D. Example: data.nii.gz
            output (str): Name of output mask, Example: data.nii.
            size: Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example:
                    35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41).
            shape (str): Shape of the mask (default: cylinder).
            remove (int): Remove temporary files (default: 1).
            verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended (default: 1).
            process (str): Process to generate mask.
                          <point,FILE>: Center mask at the X,Y coordinates of
                          the label defined in input volume FILE. (e.g.
                          "point,label.nii.gz")

        Returns:
            output (str): Output nifti file with SCT mask.
        """
    subprocess.run(['sct_create_mask', '-i', input, '-p', 'point', process, '-size', size, '-f', shape, 'o', output,
                    '-r', remove, '-v', verbose], check=True)
    click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
               % os.path.abspath(output))
    return output


@mask.command(context_settings=CONTEXT_SETTINGS, help=f"Creates a SCT (SpinalCordToolbox) mask from the input file "
                                                      f"with center process. Returns an output nifti file with "
                                                      f"SCT mask.")
@click.option('-input', type=click.File('r'), required=True, help="Complete input path of the nifti file to mask.")
@click.option('-output', type=click.Path(), default=os.curdir, help="Name of output mask, Example: data.nii.")
@click.option('-size', default='41', help="Size of the mask in the axial plane, given in pixel (Example: 35) or in "
                                          "millimeter (Example: 35mm). If shape=gaussian, size corresponds to sigma ("
                                          "Example: 45).")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask: cylinder, box or gaussian.")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="Remove temporary files.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
def sct_center(input, output, size, shape, remove, verbose):
    """
        Creates a SCT (SpinalCordToolbox) mask from the input file with center process. Returns an output nifti file
        with SCT mask.

        Args:
            input (file): Input nifti file to mask. Must be 3D. Example: data.nii.gz
            output (str): Name of output mask, Example: data.nii.
            size: Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example:
                    35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41).
            shape (str): Shape of the mask (default: cylinder).
            remove (int): Remove temporary files (default: 1).
            verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended (default: 1).


        Returns:
            output (str): Output nifti file with SCT mask.
        """
    subprocess.run(['sct_create_mask', '-i', input, '-p', 'center', '-size', size, '-f', shape, 'o', output, '-r',
                    remove, '-v', verbose], check=True)
    click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
               % os.path.abspath(output))
    return output


@mask.command(context_settings=CONTEXT_SETTINGS, help=f"Creates a SCT (SpinalCordToolbox) mask from the input file "
                                                      f"with centerline process. Returns an output nifti file with "
                                                      f"SCT mask.")
@click.option('-input', type=click.File('r'), required=True, help="Complete input path of the nifti file to mask.")
@click.option('-output', type=click.Path(), default=os.curdir, help="Name of output mask, Example: data.nii.")
@click.option('-size', default='41', help="Size of the mask in the axial plane, given in pixel (Example: 35) or in "
                                          "millimeter (Example: 35mm). If shape=gaussian, size corresponds to sigma ("
                                          "Example: 45).")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask: cylinder, box or gaussian.")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="Remove temporary files.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
@click.option('-process', type=click.File('r'), required=True,
              help='<centerline,FILE>: At each slice, the mask is centered at the spinal cord centerline, defined by '
                   'the input segmentation FILE. (e.g. "centerline,t2_seg.nii.gz")')
def sct_centerline(input, output, size, shape, remove, verbose, process):
    """
        Creates a SCT (SpinalCordToolbox) mask from the input file with centerline process. Returns an output nifti file
         with SCT mask.

        Args:
            input (file): Input nifti file to mask. Must be 3D. Example: data.nii.gz
            output (str): Name of output mask, Example: data.nii.
            size: Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example:
                    35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41).
            shape (str): Shape of the mask (default: cylinder).
            remove (int): Remove temporary files (default: 1).
            verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended (default: 1).
            process (str): Process to generate mask.
                          <centerline,FILE>: At each slice, the mask is centered
                          at the spinal cord centerline, defined by the input
                          segmentation FILE. (e.g. "centerline,t2_seg.nii.gz")
                          (default: center)

        Returns:
            output (str): Output nifti file with SCT mask.
        """
    subprocess.run(['sct_create_mask', '-i', input, '-p', 'centerline', process, '-size', size, '-f', shape, 'o',
                    output, '-r', remove, '-v', verbose], check=True)
    click.echo('The path to the output nifti file (mask.nii.gz) that contains the mask is: %s'
               % os.path.abspath(output))
    return output
