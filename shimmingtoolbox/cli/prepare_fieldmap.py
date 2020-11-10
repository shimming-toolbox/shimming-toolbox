#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-phase', 'fname_phase', type=click.Path(), required=True, help="Input path of phase nifti file")
@click.option('-mag', 'fname_mag', type=click.Path(), required=True, help="Input path of mag nifti file")
@click.option('-output', 'path_output', type=click.Path(), default=os.curdir, help="Output path for the fieldmap")
def prepare_fieldmap_cli(fname_phase, fname_mag, path_output):
    """Creates fieldmap from phase and magnitute images"""
    
    pass
