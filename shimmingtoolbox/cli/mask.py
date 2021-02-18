#!/usr/bin/env python3
6,5,4
import click
import nibabel as nib
import numpy as np
import os
import errno

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.language import English as notice
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS,
             help=notice._mask_group_help)
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=notice._mask_help)
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help=notice._mask_input_3D)
@click.option('-output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              help=notice._mask_output)           
@click.option('-size', nargs=3, required=True, type=int, help=notice._mask_box_3d)
@click.option('-center', nargs=3, type=int, default=(None, None, None),
              help=notice._mask_centre) 
def box(fname_input, output, size, center):
    nii = nib.load(fname_input)
    # convert nifti file to numpy array
    data = nii.get_fdata()
    if len(data.shape) == 3:
        mask_cb = shape_cube(data, size[0], size[1], size[2], center[0], center[1], center[2])  		
        # creation of the box mask
        mask_cb = mask_cb.astype(int)
        nii_img = nib.Nifti1Image(mask_cb, nii.affine)
        nib.save(nii_img, output)
        click.echo(notice._mask_output_filemask)
        return output
        
    else:
        raise ValueError(errno.ENODATA, notice._nifty_3d)


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=notice._mask_rectange_from_input)
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help=notice._mask_path_input_2D3D)
@click.option('-output', type=click.Path(), default=os.curdir,
              help=notice._mask_output)
@click.option('-size', nargs=2, required=True, type=int,
              help=notice._mask_box_2d)
@click.option('-center', nargs=2, type=int, default=(None, None),
              help=notice._mask_box)

def rect(fname_input, output, size, center):
    nii = nib.load(fname_input)
    # convert nifti file to numpy array
    data = nii.get_fdata()  

    if len(data.shape) == 2:
    	# creation of the rectangle mask
        mask_sqr = shape_square(data, size[0], size[1], center[0], center[1])  
        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, output)
        click.echo(notice._mask_output_filemask +"{os.path.abspath(output)}")
        return output

    elif len(data.shape) == 3:
    	# initialization of 3D array of zeros
        mask_sqr = np.zeros_like(data)  
        for z in range(data.shape[2]):
            # extraction of a MRI slice (2D)
            img_2d = data[:, :, z]  
            # creation of the mask on each slice (2D)
            mask_slice = shape_square(img_2d, size[0], size[1], center[0], center[1])
            # addition of each masked slice to form a 3D array
            mask_sqr[:, :, z] = mask_slice

        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, output)
        click.echo(notice._mask_output_filemask +"{os.path.abspath(output)}")
        return output

    else:
        raise ValueError(errno.ENODATA, notice._nifty_2d_3d)


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=notice._mask_threshold)
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help=notice._mask_input_thresh )                         
@click.option('-output', type=click.Path(), default=os.curdir,
              help=notice._mask_output)
@click.option('-thr', default=30, help=notice._mask_threshold_value )
def threshold(fname_input, output, thr):
    nii = nib.load(fname_input)
    # convert nifti file to numpy array
    data = nii.get_fdata()  

    # creation of the threshold mask
    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, output)
    click.echo(notice._mask_output_filemask+" {os.path.abspath(output)}")
    return output
