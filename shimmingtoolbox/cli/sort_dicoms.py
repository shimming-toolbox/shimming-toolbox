#!usr/bin/env python3
# coding: utf-8

import click
import logging
import os
from pydicom import dcmread
import shutil

from shimmingtoolbox.utils import set_all_loggers, create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', "--input", "path_input", help="Input folder.")
@click.option('-o', "--output", "path_output", help="Output folder.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def sort_dicoms(path_input, path_output, verbose):
    set_all_loggers(verbose)

    # Create output directory
    create_output_dir(path_output)

    # Create a list containing all the DICOMs in the input folder
    list_dicoms = []
    for name in os.listdir(path_input):
        fname_tmp = os.path.join(path_input, name)
        # If it's not a file
        if not os.path.isfile(fname_tmp):
            continue
        # If it's not a .dcm or .ima
        if (fname_tmp[-4:] != '.ima') and (fname_tmp[-4:] != '.dcm'):
            continue
        list_dicoms.append(fname_tmp)

    # Make sure there is at least one DICOM
    if not list_dicoms:
        raise RuntimeError(f"{path_input} does not contain dicom files")

    # For loop on all DICOMs in the directory
    for fname_dcm in sorted(list_dicoms):
        # Create the file path
        ds = dcmread(fname_dcm)
        series_number = ds["SeriesNumber"].value
        series_description = ds["SeriesDescription"].value
        folder_name = f"{series_number:02d}-" + series_description
        path_folder_output = os.path.join(path_output, folder_name)

        # Create output directory with the new name if it does not exist
        if not os.path.isdir(path_folder_output):
            create_output_dir(path_folder_output)

        fname_output = os.path.join(path_folder_output, os.path.basename(fname_dcm))
        # Copy files to the new folder
        shutil.copyfile(fname_dcm, fname_output)

    logger.info("Successfully sorted the DICOMs")
