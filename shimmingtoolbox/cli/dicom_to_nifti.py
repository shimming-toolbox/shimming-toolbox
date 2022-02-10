#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import os

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_config_dcm2bids__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'path_dicom', type=click.Path(), required=True, help="Input path to DICOM folder")
@click.option('-o', '--output', 'path_nifti', type=click.Path(), default=os.curdir, help="Output path to BIDS folder")
@click.option('--subject', required=True, help="Name of the imaged subject")
@click.option('--config', 'fname_config', type=click.Path(), default=__dir_config_dcm2bids__,
              help="Path to the BIDS config file")
@click.option('--remove-tmp/--dont-remove-tmp', default=False,
              help="Remove / keep tmp folder after processing")
def dicom_to_nifti_cli(path_dicom, path_nifti, subject, fname_config, remove_tmp):
    """Converts dicom files into nifti files by calling ``dcm2bids``."""

    dicom_to_nifti(path_dicom=path_dicom,
                   path_nifti=path_nifti,
                   subject_id=subject,
                   path_config_dcm2bids=fname_config,
                   remove_tmp=remove_tmp)
