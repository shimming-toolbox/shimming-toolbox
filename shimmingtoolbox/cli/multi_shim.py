#!/usr/bin/python3
# -*- coding: utf-8 -*

import click
import os
import math
import numpy as np
import nibabel as nib

from shimmingtoolbox.optimizer.lsq_optimizer import LSQ_Optimizer

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.argument('masks', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-fmap', 'fname_fmap', type=click.Path(exists=True), required=True, help="Input path of prepared fieldmap nifti file")
@click.option('-coil', 'fname_coil', type=click.Path(exists=True), required=True, help="Input path of coil array nifti file")
@click.option('-output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'shim'), help="Output filename for the fieldmap")
@click.option('-bounds', 'fname_bounds', type=click.Path(), default=None, help="Input path for bounds file")

def multi_shim_cli(masks, fname_fmap, fname_coil, fname_bounds, fname_output):
    """TODO

    masks: Input path(s) of mask nifti file(s), in ascending order: mask1, mask2, etc.
    """

    # Load
    if fname_bounds is not None:
        pass
    bounds = None

    coil = nib.load(fname_coil).get_fdata()
    fmap = nib.load(fname_fmap).get_fdata()
    
    optimizer = LSQ_Optimizer(coil)

    if os.path.isfile(fname_output + '.csv'):
        i = 0
        while os.path.isfile(fname_output + f'_{i}.csv'):
            i += 1
        fname_output += f'_{i}'
    fname_output += '.csv'

    with open(fname_output, mode='w') as f:
        for fname_mask in masks:
            mask = nib.load(fname_mask).get_fdata()
            currents = optimizer.optimize(fmap, mask, bounds=bounds)
            np.savetxt(f, currents, fmt='%.4f', newline=",")