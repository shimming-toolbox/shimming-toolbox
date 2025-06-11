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
AXES = ['-1', '0', '1', '2', '3', '4', '5', '6']
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
@click.option('--axis', type=click.Choice(AXES), default=AXES[0], show_default=True,
              help="Axis of the array to calculate the average")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def mean(fname_input, fname_output, axis, verbose):
    """Average NIfTI data across dimension."""

    # Set logger level
    set_all_loggers(verbose)

    # Load input
    nii_input = nib.load(fname_input)

    # Convert axis to integer
    axis = int(axis)

    # Make sure dimensions are appropriate
    if (nii_input.ndim - 1) < axis:
        raise ValueError(f"Axis: {axis} is out of bounds for array with {nii_input.ndim} dimensions")

    # Calculate the average
    avg = np.mean(nii_input.get_fdata(), axis=axis)

    # Create nibabel output
    nii_output = nib.Nifti1Image(avg, nii_input.affine, header=nii_input.header)

    # Save image
    if fname_output is None:
        _, filename = os.path.split(fname_input)
        fname_output = add_suffix(os.path.join(os.curdir, filename), '_mean')

    create_output_dir(fname_output, is_file=True)

    nib.save(nii_output, fname_output)


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', type=click.Path(exists=True), required=True,
              help="Input filename, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=None,
              help="Output filename, supported extensions: .nii, .nii.gz. [default: ./input_std.nii.gz]")
@click.option('--axis', type=click.Choice(AXES), default=AXES[0], show_default=True,
              help="Axis of the array to calculate the average")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def std(fname_input, fname_output, axis, verbose):
    """Compute the STD from NIfTI data across an axis."""

    # Set logger level
    set_all_loggers(verbose)

    # Load input
    nii_input = nib.load(fname_input)

    # Convert axis to integer
    axis = int(axis)

    # Make sure dimensions are appropriate
    if (nii_input.ndim - 1) < axis:
        raise ValueError(f"Axis: {axis} is out of bounds for array with {nii_input.ndim} dimensions")

    # Compute STD
    std_data = np.std(nii_input.get_fdata(), axis=axis)

    # Change the output datatype to float64
    header = nii_input.header
    header.set_data_dtype(np.float64)

    # Create nibabel output
    nii_output = nib.Nifti1Image(std_data, nii_input.affine, header=header)

    # Save image
    if fname_output is None:
        _, filename = os.path.split(fname_input)
        fname_output = add_suffix(os.path.join(os.curdir, filename), '_std')
    nib.save(nii_output, fname_output)


@maths_cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', type=click.Path(exists=True), required=True,
              help="Input filename, supported extensions: .nii, .nii.gz")
@click.option('-d', '--denominator', 'fname_denom', type=click.Path(exists=True), required=True,
              help="Input filename, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=None,
              help="Output filename, supported extensions: .nii, .nii.gz. [default: ./input_div.nii.gz]")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def div(fname_input, fname_denom, fname_output, verbose):
    """Divide NIfTI input by NIfTI input 2."""

    # Set logger level
    set_all_loggers(verbose)

    # Load input
    nii_input = nib.load(fname_input)
    nii_denom = nib.load(fname_denom)

    if nii_input.ndim < nii_denom.ndim:
        raise ValueError(f"Input image {fname_input} has fewer dimensions than denominator image {fname_denom}. "
                         f"Cannot perform division.")

    # Compute division
    div_data = nii_input.get_fdata() / nii_denom.get_fdata()

    # Create nibabel output
    nii_output = nib.Nifti1Image(div_data, nii_input.affine, header=nii_input.header)

    # Save image
    if fname_output is None:
        _, filename = os.path.split(fname_input)
        fname_output = add_suffix(os.path.join(os.curdir, filename), '_div')
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
        raise ValueError("The options --real and --imaginary, or --complex should be provided.")

    # Create nibabel output
    header = nii.header
    header['datatype'] = 16  # 16 is float32
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
        raise ValueError("The options --real and --imaginary, or --complex should be provided.")

    # Create nibabel output
    header = nii.header
    header['datatype'] = 16  # 16 is float32
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
