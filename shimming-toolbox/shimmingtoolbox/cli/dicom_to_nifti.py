#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import logging
import os

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __config_dcm2bids__
from shimmingtoolbox.utils import set_all_loggers

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'path_dicom', type=click.Path(), required=True,
              help="Input path to DICOM folder")
@click.option('--subject', required=True, help="Name of the imaged subject")
@click.option('-o', '--output', 'path_nifti', type=click.Path(), default=os.curdir,
              help="Output path to BIDS folder")
@click.option('--config', 'fname_config', type=click.Path(), default=__config_dcm2bids__, show_default=True,
              help="Path to dcm2bids config file")
@click.option('--rm-tmp', 'remove_tmp', is_flag=True, help="Remove tmp folder")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
def dicom_to_nifti_cli(path_dicom, subject, path_nifti, fname_config, remove_tmp, verbose):
    """Converts DICOM files into NIfTI files by calling ``dcm2bids``."""

    # Set logger level
    set_all_loggers(verbose)

    dicom_to_nifti(path_dicom=path_dicom,
                   path_nifti=path_nifti,
                   subject_id=subject,
                   fname_config_dcm2bids=fname_config,
                   remove_tmp=remove_tmp)
