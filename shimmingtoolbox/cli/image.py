#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import os
import nibabel as nib
import logging
import numpy as np

from shimmingtoolbox.image import concat_data

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
AXES = ['0', '1', '2', '3', '4']


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Perform manipulations on images")
def image_cli():
    pass


@image_cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('input', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'concat.nii.gz'),
              show_default=True, help="Output filename, supported extensions: .nii, .nii.gz")
@click.option('--axis', type=click.Choice(AXES), default=AXES[3], show_default=True,
              help="Dimension of the array to concatenate")
@click.option('--pixdim', type=click.FLOAT, help="Pixel resolution to join to image header")
def concat(input, axis, fname_output, pixdim):
    """Concatenate NIfTIs along the specified dimension.

    INPUT: Input paths of the files to concatenate. Separate the files by a space.
    """
    # Create nii list
    list_nii = []
    for fname_file in input:
        nii_input = nib.load(fname_file)
        list_nii.append(nii_input)

    # Call concat API
    dim_list = AXES
    index = dim_list.index(axis)
    nii_out = concat_data(list_nii, index, pixdim)

    # Save image
    nib.save(nii_out, fname_output)


@image_cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('inputs', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(os.curdir, 'logical_and.nii.gz'), show_default=True,
              help="Output filename, supported extensions: .nii, .nii.gz")
def logical_and(inputs, fname_output):
    """Calculate the logical and for a number of NIfTIs

    INPUTS: Input paths of the files to apply the logical and. Separate the files by a space.
    """

    if len(inputs) == 1:
        logger.info("Only 1 file provided, output is the same as the input.")
        nii = nib.load(inputs[0])
        nib.save(nii, fname_output)
        return 0

    # Create nii list
    list_nii = []
    dimensions = nib.load(inputs[0]).shape
    affine = nib.load(inputs[0]).affine
    for fname_file in inputs:
        nii_input = nib.load(fname_file)
        # Make sure dimensions and affines are the same
        if not np.all(nii_input.shape == dimensions):
            raise ValueError("Dimensions of all NIfTIs must be the same")
        if not np.all(nii_input.affine == affine):
            raise ValueError("Affine of all NIfTIs must be the same")
        list_nii.append(nii_input)

    # Apply the logical and
    output = np.full(dimensions, True)
    for nii in list_nii:
        output = np.logical_and(output, nii.get_fdata())

    # Save image
    nii_out = nib.Nifti1Image(output, nii_input.affine, header=nii_input.header)
    nib.save(nii_out, fname_output)
