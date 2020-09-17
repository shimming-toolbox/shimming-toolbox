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
    "testing_data": (["https://github.com/shimming-toolbox/data-testing/archive/r20200713.zip"], "Light-weighted dataset for testing purpose."),
    "prelude": (["https://github.com/shimming-toolbox/binaries/raw/master/prelude"], "Binary for prelude software")
}

dataset_list_str: str = ""

for item in URL_DICT.items():
    dataset_list_str += f"\n\n - {item[0]}: {item[1][1]}"

@click.command(help=f"Download data from the internet. The available datasets are:{dataset_list_str}")
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.option("--output", help="Output folder.")
@click.argument("data")
def main(verbose, output, data):
    """
    Download data from the internet.

    Args:
        verbose: If true, increases output verbosity.
        output: Output folder.
        data: The data to be downloaded.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f'{output}, {data}')
    url = URL_DICT[data][0]
    if output is None:
        output = os.path.join(os.path.abspath(os.curdir), data)
    install_data(url, output, keep=True)


if __name__ == '__main__':
    main()
