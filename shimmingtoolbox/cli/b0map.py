#!/usr/bin/env python3

import logging

import click

from .. import b0map


@click.command()
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.option("--coilshims", help="Output hardware-shim calibration.")
@click.option("--pulseseq", help="Output pulse-sequence calibration.")
@click.argument("input", type=click.Path(exists=True, file_okay=False))
@click.argument("output", type=click.Path(exists=False))
def main(verbose, coilshims, pulseseq, input, output):
    """
    Compute b0 fieldmap ``OUTPUT`` from a folder of dicom files ``INPUT/``.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f"{coilshims}, {pulseseq}, {input}, {output}")
    fieldmap, coilshims, pulseseq = b0map(input, coilshims=coilshims, pulseseq=pulseseq)
    with open(output, "w") as out:
        out.write(str(fieldmap))  # definitely not the right file format


if __name__ == "__main__":
    main()
