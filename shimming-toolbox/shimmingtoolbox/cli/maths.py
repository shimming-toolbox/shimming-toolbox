#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
from cloup import command, option, option_group, group
from cloup.constraints import RequireAtLeast, mutually_exclusive, all_or_none, constraint
import json
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.utils import add_suffix, set_all_loggers, splitext, save_nii_json, create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
AXES = ['0', '1', '2', '3', '4']
DEFAULT_PATH = os.path.abspath(os.curdir)


@group(context_settings=CONTEXT_SETTINGS,
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


@command(context_settings=CONTEXT_SETTINGS)
@option_group("Inputs",
              option('--real', 'fname_real', type=click.Path(exists=True), required=False,
                     help="Input filename of a real image. Must be used with '--imaginary'. "
                          "Supported extensions: .nii, .nii.gz"),
              option('--imaginary', 'fname_im', type=click.Path(exists=True), required=False,
                     help="Input filename of a imaginary image. Must be used with '--real'. "
                          "Supported extensions: .nii, .nii.gz"),
              option('--complex', 'fname_complex', type=click.Path(exists=True), required=False,
                     help="Input filename of a complex image. Supported extensions: .nii, .nii.gz"),
              constraint=RequireAtLeast(1))
@constraint(all_or_none, ['fname_real', 'fname_im'])
@constraint(mutually_exclusive, ['fname_real', 'fname_complex'])
@constraint(mutually_exclusive,['fname_im', 'fname_complex'])
@option('-o', '--output', 'fname_output', type=click.Path(),
                     default=os.path.join(DEFAULT_PATH, 'phase.nii.gz'),
                     help="Output filename. Supported extensions: .nii, .nii.gz")
@option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def phase(fname_im, fname_real, fname_complex, fname_output, verbose):
    """Compute the phase data from real and imaginary data, or complex data."""

    # Set logger level
    set_all_loggers(verbose)

    fname_output = os.path.abspath(fname_output)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Load input
    if fname_complex is not None:
        nii = nib.load(fname_complex)
        # Calculate the phase
        phase_data = np.angle(np.asanyarray(nii.dataobj))
        fname_json = splitext(fname_complex)[0] + '.json'
    elif fname_real is not None and fname_im is not None:
        nii = nib.load(fname_real)
        nii_im = nib.load(fname_im)
        # Calculate the phase
        phase_data = np.arctan2(nii_im.get_fdata(), nii.get_fdata())
        fname_json = splitext(fname_real)[0] + '.json'
    else:
        raise ValueError("At least one of the inputs must be provided.")

    # Create nibabel output
    nii_output = nib.Nifti1Image(phase_data, nii.affine, header=nii.header)

    # Save nii and JSON if possible
    if os.path.isfile(fname_json):
        with open(fname_json, 'r') as json_file:
            json_data = json.load(json_file)

        if 'ImageType' in json_data:
            for i_field, field in enumerate(json_data['ImageType']):
                if 'M' == field.upper() or 'R' == field.upper() or 'I' == field.upper() or 'C' == field.upper():
                    json_data['ImageType'][i_field] = 'P'
                if ('MAG' == field.upper() or 'REAL' == field.upper() or 'IMAGINARY' == field.upper() or
                        'COMPLEX' == field.upper()):
                    json_data['ImageType'][i_field] = 'PHASE'

        save_nii_json(nii_output, json_data, fname_output)
    else:
        nib.save(nii_output, fname_output)


@command(context_settings=CONTEXT_SETTINGS)
@option_group("Inputs",
              option('--real', 'fname_real', type=click.Path(exists=True), required=False,
                     help="Input filename of a real image. Must be used with '--imaginary'. "
                          "Supported extensions: .nii, .nii.gz"),
              option('--imaginary', 'fname_im', type=click.Path(exists=True), required=False,
                     help="Input filename of a imaginary image. Must be used with '--real'. "
                          "Supported extensions: .nii, .nii.gz"),
              option('--complex', 'fname_complex', type=click.Path(exists=True), required=False,
                     help="Input filename of a complex image. Supported extensions: .nii, .nii.gz"),
              constraint=RequireAtLeast(1))
@constraint(all_or_none, ['fname_real', 'fname_im'])
@constraint(mutually_exclusive, ['fname_real', 'fname_complex'])
@constraint(mutually_exclusive,['fname_im', 'fname_complex'])
@option('-o', '--output', 'fname_output', type=click.Path(),
                     default=os.path.join(DEFAULT_PATH, 'phase.nii.gz'),
                     help="Output filename. Supported extensions: .nii, .nii.gz")
@option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def mag(fname_im, fname_real, fname_complex, fname_output, verbose):
    """Compute the magnitude data from real and imaginary data, or complex data."""

    # Set logger level
    set_all_loggers(verbose)

    fname_output = os.path.abspath(fname_output)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Load inputs
    if fname_complex is not None:
        nii = nib.load(fname_complex)
        # Calculate the magnitude
        mag = np.abs(np.asanyarray(nii.dataobj))
        fname_json = splitext(fname_complex)[0] + '.json'
    elif fname_real is not None and fname_im is not None:
        nii = nib.load(fname_real)
        nii_im = nib.load(fname_im)
        # Calculate the magnitude
        mag = np.sqrt(nii_im.get_fdata() ** 2 + nii.get_fdata() ** 2)
        fname_json = splitext(fname_real)[0] + '.json'
    else:
        raise ValueError("At least one of the inputs must be provided.")

    # Create nibabel output
    nii_output = nib.Nifti1Image(mag, nii.affine, header=nii.header)

    # Save nii and JSON if possible
    if os.path.isfile(fname_json):
        with open(fname_json, 'r') as json_file:
            json_data = json.load(json_file)

        if 'ImageType' in json_data:
            for i_field, field in enumerate(json_data['ImageType']):
                if 'P' == field.upper() or 'R' == field.upper() or 'PHASE' == field.upper() or 'REAL' == field.upper()\
                        or 'I' == field.upper() or 'IMAGINARY' == field.upper() or 'COMPLEX' == field.upper() \
                        or 'C' == field.upper():
                    json_data['ImageType'][i_field] = 'MAGNITUDE'

        save_nii_json(nii_output, json_data, fname_output)
    else:
        nib.save(nii_output, fname_output)


maths_cli.add_command(mag)
maths_cli.add_command(phase)
