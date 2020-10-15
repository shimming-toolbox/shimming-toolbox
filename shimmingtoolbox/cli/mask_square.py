#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np

from shimmingtoolbox.masking.shapes import shape_square


@click.command()
@click.argument("input", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("-len_dim1", required=True, type=int, help="Length of the side of the square along first dimension (in "
                                                         "pixels)")
@click.option("-len_dim2", required=True, type=int, help="Length of the side of the square along second dimension (in "
                                                         "pixels)")
@click.option("-center_dim", nargs=2, type=int, help="Center of the square along first and second dimension (in "
                                                     "pixels). If no center is provided, the middle is used.")
def mask_square(input, len_dim1, len_dim2, center_dim):
    """
            Apply a square mask to the input file. Returns mask.

            Args:
                input (str): Data to mask.
                len_dim1 (int): Length of the side of the square along first dimension (in pixels).
                len_dim2 (int): Length of the side of the square along second dimension (in pixels).
                center_dim (int): Center of the square along first and second dimension (in pixels). If no center is
                                provided, the middle is used.

            Return:
                mask_sqr (numpy.ndarray): Mask with booleans.
            """
    im_nii = nib.load(input)
    nii_data = im_nii.get_fdata()  # convert nifty file to numpy array

    if len(nii_data.shape) == 2:
        mask_sqr = shape_square(nii_data, len_dim1, len_dim2, center_dim)  # application of the mask
        return mask_sqr

    elif len(nii_data.shape) == 3:
        for z in range(nii_data.shape[2]):
            mask_sqr = np.zeros_like(nii_data)  # initialization of 3D array of zeros
            img_2d = nii_data[:, :, z]  # extraction of a MRI slice (2D)
            mask = shape_square(img_2d, len_dim1, len_dim2, center_dim)  # application of the mask on each slice (2D)
            mask_sqr[:, :, z] = mask  # addition of each masked slice to form a 3D array
            return mask_sqr

    else:
        return None
