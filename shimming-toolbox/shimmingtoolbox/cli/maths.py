#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.utils import add_suffix, set_all_loggers, splitext, save_nii_json, create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
AXES = ['0', '1', '2', '3', '4']
DEFAULT_PATH = os.path.abspath(os.curdir)


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Perform mathematical operations on images.")
def maths_cli():
    pass


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', type=click.Path(exists=True), required=True,
              help="Input filename, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=None,
              help="Output filename, supported extensions: .nii, .nii.gz. [default: ./input_mean.nii.gz]")
@click.option('--axis', type=click.Choice(AXES), default=AXES[3], show_default=True,
              help="Axis of the array to calculate the average")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def mean(fname_input, fname_output, axis, verbose):
    """Average NIfTI data across dimension."""

    # Set logger level
    set_all_loggers(verbose)

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


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--imaginary', 'fname_im', type=click.Path(exists=True), required=True,
              help="Input filename of a imaginary image, supported extensions: .nii, .nii.gz")
@click.option('--real', 'fname_real', type=click.Path(exists=True), required=True,
              help="Input filename of a real image, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(DEFAULT_PATH, 'phase.nii.gz'),
              help="Output filename, supported extensions: .nii, .nii.gz")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def phase(fname_im, fname_real, fname_output, verbose):
    """Compute the phase data from other image types."""

    # Set logger level
    set_all_loggers(verbose)

    fname_output = os.path.abspath(fname_output)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Load input
    nii_real = nib.load(fname_real)
    nii_im = nib.load(fname_im)

    # Calculate the phase
    phase = np.arctan2(nii_im.get_fdata(), nii_real.get_fdata())

    # Create nibabel output
    nii_output = nib.Nifti1Image(phase, nii_real.affine, header=nii_real.header)

    # Save nii and JSON if possible
    fname_json_real = splitext(fname_real)[0] + '.json'
    if os.path.isfile(fname_json_real):
        with open(fname_json_real, 'r') as json_file:
            json_data = json.load(json_file)

        save_nii_json(nii_output, json_data, fname_output)
    else:
        nib.save(nii_output, fname_output)


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--imaginary', 'fname_im', type=click.Path(exists=True), required=True,
              help="Input filename of a imaginary image, supported extensions: .nii, .nii.gz")
@click.option('--real', 'fname_real', type=click.Path(exists=True), required=True,
              help="Input filename of a real image, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(DEFAULT_PATH, 'phase.nii.gz'),
              help="Output filename, supported extensions: .nii, .nii.gz")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def mag(fname_im, fname_real, fname_output, verbose):
    """Compute the magnitude data from other image types."""

    # Set logger level
    set_all_loggers(verbose)

    fname_output = os.path.abspath(fname_output)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Load input
    nii_real = nib.load(fname_real)
    nii_im = nib.load(fname_im)

    # Calculate the phase
    mag = np.sqrt(nii_im.get_fdata() ** 2 + nii_real.get_fdata() ** 2)

    # Create nibabel output
    nii_output = nib.Nifti1Image(mag, nii_real.affine, header=nii_real.header)

    # Save nii and JSON if possible
    fname_json_real = splitext(fname_real)[0] + '.json'
    if os.path.isfile(fname_json_real):
        with open(fname_json_real, 'r') as json_file:
            json_data = json.load(json_file)

        save_nii_json(nii_output, json_data, fname_output)
    else:
        nib.save(nii_output, fname_output)
