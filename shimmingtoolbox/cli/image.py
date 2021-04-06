#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os
import nibabel as nib

from shimmingtoolbox.image import concat_data

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
AXES = ['0', '1', '2', '3', '4']


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Perform manipulations on images")
def image_cli():
    pass


@image_cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('input', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'concat.nii.gz'),
              help="Output filename, supported extensions: .nii, .nii.gz")
@click.option('--axis', type=click.Choice(AXES), required=True, help="Dimension of the array to concatenate")
@click.option('--pixdim', type=click.FLOAT, required=False, help="Pixel resolution to join to image header")
def concat(input, axis, fname_output, pixdim):
    """Concatenate NIfTIs along the specified dimension

    input: Input paths for the files to concatenate. Separate the files by a space.
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
