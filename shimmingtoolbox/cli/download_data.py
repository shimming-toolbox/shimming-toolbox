#!/usr/bin/env python3

import os
import logging
import click

from shimmingtoolbox.download import install_data


dict_url = {
    "testing_data":
        ["https://github.com/shimming-toolbox/data-testing/archive/r20200713.zip",
    ]
}

# TODO: display automatically the list of data avaialble from the dict above
# TODO: wrap the help properly
@click.command(help="Download data from the internet. The available datasets are:"
                    "- testing_data: Light-weighted dataset for testing purpose.")
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.option("--output", help="Output folder.")
@click.argument("data")
def main(verbose, output, data):
    """
    Download data from the internet.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f'{output}, {data}')
    url = dict_url[data]
    if output is None:
        output = os.path.join(os.path.abspath(os.curdir), data)
    install_data(url, output)


if __name__ == '__main__':
    main()
