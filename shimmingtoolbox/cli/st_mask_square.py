#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np

from shimmingtoolbox.masking.shapes import shape_square


@click.command()
@click.argument("filename", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--len_dim1", '-l1', required=True, type=click.IntRange(0, 255, clamp=True), help="Length of the side "
                                                                                                "of the square along "
                                                                                                "third dimension (in "
                                                                                                "pixels)")
@click.option("--len_dim2", '-l2', required=True, type=click.IntRange(0, 255, clamp=True), help="Length of the side "
                                                                                                "of the square along "
                                                                                                "third dimension (in "
                                                                                                "pixels)")
@click.option("--center_dim", nargs=2, default=0, help="Center of the square along first and second dimension (in "
                                                       "pixels). If no center is provided, the middle is used.")
def st_mask_cube(filename, len_dim1, len_dim2, center_dim):
    """
            Apply a square mask to the input file. Returns mask.

            Args:
                filename (str): Data to mask.
                len_dim1 (int): Length of the side of the square along first dimension (in pixels).
                len_dim2 (int): Length of the side of the square along second dimension (in pixels).
                center_dim (int): Center of the square along first and second dimension (in pixels). If no center is
                                provided, the middle is used.

            Return:
                mask_square (numpy.ndarray): Mask with booleans.
            """
    img_nifti = nib.load(filename)
    data = img_nifti.get_fdata()  # convert nifty file to numpy array
    mask_square = np.array([[[]]])  # initialization of 3D array

    for z in range(data.shape[2]):
        img_2d = data[:, :, z]  # extraction of a MRI slice
        mask = shape_square(img_2d, len_dim1, len_dim2, center_dim)  # application of the mask on each slice
        mask_square.append(mask)  # addition of each masked slice to form a 3D array
        # problem with np.append & boolean mask

    return mask_square
