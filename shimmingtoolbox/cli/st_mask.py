#!/usr/bin/env python3

import logging
import click
import numpy as np

from shimmingtoolbox.masking import threshold
from shimmingtoolbox.masking import shapes
# import SCT
# import BET

@click.command()
@click.option("--verbose", is_flag=True, help="Be more verbose.")
@click.argument("input", required=True, type=click.File('rb')) # 1 file à la fois ou bien plusieurs d'un coup avec nargs=-1 ?
@click.option("--method", required=True, type=click.Choice(['SCT','BET','shape','threshold'], case_sensitive=False), help="Indicate which kind od mask: SCT, BET, shape or threshold")
@click.option("--shape", type=click.Choice(['cube','square'], case_sensitive=False), default='cube', help="Indcate which kinf of shape for method=shape: cube or square")
@click.option("--size", type=int, default=1, help="Radius of the shape in voxel")
def main(verbose, input, method, shape, size):
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info(f'{input}, {method}, {shape}, {size}')

    # data = np.fromfile(input, dtype=?)  --> Pour écrire le binary file en array ?

    if method == 'shape':
        st_mask = shapes(input,shape,size) # size en voxel alors que taille attendue en pixel dans cette fonction
        # input à changer en data ?
    
    elif method == 'threshold':
        st_mask = threshold(input,size) # Pourquoi pas size pour threshold ? Celle-ci est demandée en voxel dans la fonction threshold
        # input à changer en data ?
    
    # elif à venir avec SCT et BET
    
    return st_mask


if __name__ == '__main__':
    main()