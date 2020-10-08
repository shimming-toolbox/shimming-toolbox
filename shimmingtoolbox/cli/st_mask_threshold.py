#!/usr/bin/env python3

import click
import nibabel as nib

from shimmingtoolbox.masking.threshold import threshold


@click.command()
@click.argument("filename", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--thr", default=30, help="Value to threshold the data: voxels will be set to zero if their value is "
                                        "equal or less than this threshold")
def st_mask_threshold(filename, thr):
    """
        Apply a threshold mask to the input file. Returns mask.

        Args:
            filename (str): Data to be masked
            thr: Value to threshold the data: voxels will be set to zero if their
                value is equal or less than this threshold

        Returns:
            mask_thr (numpy.ndarray): Boolean mask with same dimensions as data
        """
    img_nifti = nib.load(filename)
    data = img_nifti.get_fdata()  # convert nifty file to numpy array

    mask_thr = threshold(data, thr)

    return mask_thr
