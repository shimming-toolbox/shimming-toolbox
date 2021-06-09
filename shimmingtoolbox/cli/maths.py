#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import os
import nibabel as nib
import numpy as np

from shimmingtoolbox.utils import add_suffix

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
AXES = ['0', '1', '2', '3', '4']


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Perform mathematical operations on images.")
def maths_cli():
    pass


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', 'fname_input', type=click.Path(exists=True), required=True,
              help="Input filename, supported extensions: .nii, .nii.gz")
@click.option('--output', 'fname_output', type=click.Path(), default=None,
              help="Output filename, supported extensions: .nii, .nii.gz")
@click.option('--axis', type=click.Choice(AXES), default=AXES[3], show_default=True,
              help="Axis of the array to calculate the average  [default: ./input_mean.nii.gz]")
def mean(fname_input, fname_output, axis):
    """Average NIfTI data across dimension."""

    # Load input
    nii_input = nib.load(fname_input)

    # Find index
    dim_list = AXES
    index = dim_list.index(axis)
    if len(nii_input.shape) < index:
        raise IndexError(f"Axis: {axis} is out of bounds for array of length: {len(nii_input.shape)}")

    # Calculate the average
    avg = np.mean(nii_input.get_fdata(), axis=index)

    # Create nibabel output
    nii_output = nib.Nifti1Image(avg, nii_input.affine, header=nii_input.header)

    # Save image
    if fname_output is None:
        _, filename = os.path.split(fname_input)
        fname_output = add_suffix(os.path.join(os.curdir, filename), '_mean')
    nib.save(nii_output, fname_output)
