#!/usr/bin/env python3

import logging
import click
import numpy as np

from shimmingtoolbox.masking import threshold
from shimmingtoolbox.masking import shapes
# import SCT
# import BET

@click.command()
@click.argument("input", required=True, type=click.File('rb')) # 1 file à la fois ou bien plusieurs d'un coup avec nargs=-1 ?
@click.option("--method", required=True, type=click.Choice(['SCT','BET','shape','threshold'], case_sensitive=False), help="Indicate which kind od mask: SCT, BET, shape or threshold")
@click.option("--shape", type=click.Choice(['cube','square'], case_sensitive=False), default='cube', help="Indcate which kinf of shape for method=shape: cube or square")
@click.option("--size", type=int, default=10, help="Radius of the shape in voxel")
def main(input, method, shape, size):
    """
    Apply a s-t mask to the input file. Returns mask with the same method as `method`.

    Args:
        input (binary file): Data to mask.
        method (str): Length of the side of the square along first dimension (in pixels).
        shape (str): Shape (cube or square) used when method is shape. If no shape is provided, the cube is used.
        size (int): Radius of the shape when method is shape (in voxel). If no size is provided, size=10.
        
    Returns:
        numpy.ndarray: Mask with booleans.
    """
    
    # data = np.fromfile(input, dtype=?)  #
    # To write binary file in array ?

    if method == 'shape':
        st_mask = shapes(input,shape,size) # size in voxel while expected in pixel in the function
        # `data` instead of `input` ?
    
    elif method == 'threshold':
        st_mask = threshold(input,size) # Why not size for threshold ? Size expected in voxel in this function
        # `data` instead of `input` ?
    
    # elif to come w/ SCT & BET
    
    return st_mask


if __name__ == '__main__':
    main()