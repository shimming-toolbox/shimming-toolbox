#!usr/bin/env python3
# coding: utf-8

import click
import logging
import os
from pydicom import dcmread
from pydicom.misc import is_dicom
import shutil

from shimmingtoolbox.utils import set_all_loggers, create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_RECURSIVE_DEPTH = 5


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', "--input", "path_input", help="Input folder.")
@click.option('-r', "--recursive", 'is_recursive', is_flag=True, default=False, show_default=True,
              help=f"Specifies to look into sub-folders. See also the --recursive-depth option.")
@click.option('--recursive-depth', 'recursive_depth', type=click.INT, default=DEFAULT_RECURSIVE_DEPTH,
              show_default=True, help="Depth of the recursive search.")
@click.option('-o', "--output", "path_output", help="Output folder.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def sort_dicoms(path_input, is_recursive, recursive_depth, path_output, verbose):
    set_all_loggers(verbose)

    list_dicoms = get_dicom_paths(path_input, is_recursive, 0, max_depth=recursive_depth)

    # Make sure there is at least one DICOM
    if not list_dicoms:
        raise RuntimeError(f"{path_input} does not contain dicom files, use the -r option to look into sub-folders.")

    # Create output directory
    create_output_dir(path_output)

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


def get_dicom_paths(path, is_recursive, subfolder_depth=0, max_depth=DEFAULT_RECURSIVE_DEPTH):
    # Create a list containing all the DICOMs in the input folder

    list_dicoms = []
    for name in os.listdir(path):
        fname_tmp = os.path.join(path, name)
        # If it's not a file
        if not os.path.isfile(fname_tmp):
            if os.path.isdir(fname_tmp):
                if is_recursive and subfolder_depth < max_depth:
                    list_dicoms += get_dicom_paths(fname_tmp, is_recursive, subfolder_depth + 1, max_depth=max_depth)
            continue
        # If it's not a DICOM file
        if not is_dicom(fname_tmp):
            continue

        list_dicoms.append(fname_tmp)

    return list_dicoms
