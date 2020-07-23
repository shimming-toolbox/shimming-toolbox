#!/usr/bin/env python3

import logging

import click

from .. import referencemaps

@click.command()
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.argument("inputs", required=True, nargs=-1, type=click.File('rb'))
@click.argument("output", type=click.Path(exists=False))
def main(verbose, inputs, output):
    """
    Compute a reference field b0 fieldmap OUTPUT from a folder of dicom files INPUT.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f'{inputs}, {output}')
    # TODO: do some basic type-checking here, on the given files, before feeding the library function,
    # so that you can give the user good feedback when they make a mistake.
    map = referencemaps(*inputs)
    with open(output, "w") as out:
        out.write(str(map)) # this is probably not the right format

if __name__ == '__main__':
    main()
