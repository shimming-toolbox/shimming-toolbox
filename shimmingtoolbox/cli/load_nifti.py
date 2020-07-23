#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import click

from shimmingtoolbox.load_nifti import load_nifti

@click.command(help="Load nifti data type acquisiiton from dcm2bids."
                    "- file_path: Path to the nifti type data file.")
@click.option("--verbose", is_flag=True, help="Be more verbose")
@click.argument("file_path")
def main(verbose, file_path):
    """
    Load nifti data type from file file_path.
    :param verbose: verbose funcitonality
    :param file_path: file_path of the nifti data
    :return:
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading {}".format(file_path))
    load_nifti(file_path)
    logging.info("Loading finished")

if __name__=="__main__":
    main()
