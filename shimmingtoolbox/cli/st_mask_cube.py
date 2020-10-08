#!/usr/bin/env python3

import click
import nibabel as nib

from shimmingtoolbox.masking.shapes import shape_cube


@click.command()
@click.argument("filename", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--len_dim1", '-l1', required=True, type=click.IntRange(0, 255, clamp=True), help="Length of the side "
                                                                                                "of the cube along "
                                                                                                "third dimension (in "
                                                                                                "pixels)")
@click.option("--len_dim2", '-l2', required=True, type=click.IntRange(0, 255, clamp=True), help="Length of the side "
                                                                                                "of the cube along "
                                                                                                "third dimension (in "
                                                                                                "pixels)")
@click.option("--len_dim3", '-l3', required=True, type=click.IntRange(0, 127, clamp=True), help="Length of the side "
                                                                                                "of the cube along "
                                                                                                "third dimension (in "
                                                                                                "pixels)")
@click.option("--center_dim", nargs=3, default=0, help="Center of the cube along first, second and third dimension "
                                                       "(in pixels). If no center is provided, the middle is used.")
def st_mask_cube(filename, len_dim1, len_dim2, len_dim3, center_dim):
    """
        Apply a cube mask to the input file. Returns mask.

        Args:
            filename (str): Data to mask.
            len_dim1 (int): Length of the side of the cube along first dimension (in pixels).
            len_dim2 (int): Length of the side of the cube along second dimension (in pixels).
            len_dim3 (int): Length of the side of the cube along third dimension (in pixels).
            center_dim (int): Center of the cube along first, second and third dimension (in pixels). If no center is
                            provided, the middle is used.

        Return:
            mask_cube (numpy.ndarray): Mask with booleans. True where the cube is located and False in the background.
        """
    img_nifti = nib.load(filename)
    data = img_nifti.get_fdata()  # convert nifty file to numpy array

    mask_cube = shape_cube(data, len_dim1, len_dim2, len_dim3, center_dim)

    return mask_cube
