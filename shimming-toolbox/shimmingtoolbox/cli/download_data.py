#!/usr/bin/env python3

import os
import logging
import click
from typing import Dict, Tuple, List

from shimmingtoolbox.download import install_data

# URL dictionary is in the format:
# - key: name of item to download
# - value: tuple containing:
#     0. List containing the item's URL string
#     1. String description of the item

URL_DICT: Dict[str, Tuple[List[str], str]] = {
    "testing_data": (
        ["https://github.com/shimming-toolbox/data-testing/archive/r20230929.zip"],
        "Light-weighted dataset for testing purpose.",
    ),
    "prelude": (
        ["https://github.com/shimming-toolbox/binaries/raw/master/prelude"],
        "Binary for prelude software",
    ),
    "fsl": (
        ["https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py"],
        "FSL software",
    ),
    "data_dynamic_shimming": (
        ["https://github.com/shimming-toolbox/data-dynamic-shimming/archive/r20220319.zip"],
        "B0 dynamic shimming dataset.",
    ),
    "data_create_coil_profiles": (
        ["https://github.com/shimming-toolbox/data-create-coil-profiles/releases/download/r20230815/dicoms.zip"],
        ["https://github.com/shimming-toolbox/data-create-coil-profiles/archive/refs/tags/r20230929.zip"],
        "B0 coil profile dataset.",
    ),
    "data_b1_shimming": (
        ["https://github.com/shimming-toolbox/b1_dicoms_brain/archive/r20220319.zip"],
        "Static B1+ shimming dataset.",
    ),
}

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

dataset_list_str: str = ""

for item in URL_DICT.items():
    dataset_list_str += f"\n\n - {item[0]}: {item[1][1]}"


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help=f"Download data from the internet.\n"
         f"\nDATA: The data to be downloaded. Select a dataset from the list: {dataset_list_str}"
)
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.option('-o', "--output", help="Output folder.")
@click.argument("data")
def download_data(verbose, output, data):
    """
    Download data from the internet.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f'{output}, {data}')

    if output is None:
        output = os.path.join(os.path.abspath(os.curdir), data)

    if data == 'data_create_coil_profiles':
        install_data(URL_DICT[data][0], os.path.join(output, 'dicoms'), keep=True)
        install_data(URL_DICT[data][1], output, keep=True)
    else:
        install_data(URL_DICT[data][0], output, keep=True)
