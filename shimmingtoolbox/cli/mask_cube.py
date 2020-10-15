#!/usr/bin/env python3

import click
import nibabel as nib

from shimmingtoolbox.masking.shapes import shape_cube


@click.command()
@click.argument("input", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("-len_dim1", required=True, type=int, help="Length of the side of the cube along first dimension (in "
                                                         "pixels)")
@click.option("-len_dim2", required=True, type=int, help="Length of the side of the cube along second dimension (in "
                                                         "pixels)")
@click.option("-len_dim3", required=True, type=int, help="Length of the side of the cube along third dimension (in "
                                                         "pixels)")
@click.option("-center_dim", nargs=3, type=int, help="Center of the cube along first, second and third dimension (in"
                                                     "pixels). If no center is provided, the middle is used.")
def mask_cube(input, len_dim1, len_dim2, len_dim3, center_dim):
    """
        Apply a cube mask to the input file. Returns mask.

        Args:
            input (str): Data to mask.
            len_dim1 (int): Length of the side of the cube along first dimension (in pixels).
            len_dim2 (int): Length of the side of the cube along second dimension (in pixels).
            len_dim3 (int): Length of the side of the cube along third dimension (in pixels).
            center_dim (int): Center of the cube along first, second and third dimension (in pixels). If no center is
                            provided, the middle is used.

        Return:
            mask_cb (numpy.ndarray): Mask with booleans. True where the cube is located and False in the background.
        """
    im_nii = nib.load(input)
    nii_data = im_nii.get_fdata()  # convert nifty file to numpy array

    if len(nii_data.shape) == 3:
        mask_cb = shape_cube(nii_data, len_dim1, len_dim2, len_dim3, center_dim)
        return mask_cb

    else:
        return None
