#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube
from shimmingtoolbox.utils import run_subprocess

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS,
             help=f"Create a mask based on a specified shape (box, rectangle, SpinalCord Toolbox mask) or based on the "
                  f"thresholding of an input image. Callable with the prefix 'st' in front of 'mask'. "
                  f"(Example: 'st_mask -h').")
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a box mask from the input file. The nifti file is converted to a numpy array. If this "
                       f"array is in 3D dimensions, then a binary mask is created from this array in the form of a box"
                       f" with lengths defined in 'size'. This box is centered according to the 3 dimensions indicated"
                       f" in 'center'. The mask is stored by default under the name 'mask.nii.gz' in the output folder."
                       f"Return the filename for the output mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. This nifti file must have 3D. Supported extensions are"
                   " .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-size', nargs=3, required=True, type=int,
              help="(int): Length of the side of the box along first, second and third dimension (in pixels). "
                   "(nargs=3)")
@click.option('-center', nargs=3, type=int, default=(None, None, None),
              help="(int): Center of the box along first, second and third dimension (in pixels). If no center "
                   "is provided (None), the middle is used. (nargs=3) (default: None, None, None)")
def box(fname_input, output, size, center):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 3:
        mask_cb = shape_cube(data, size[0], size[1], size[2], center[0], center[1], center[2])  # creation
        # of the box mask
        mask_cb = mask_cb.astype(int)
        nii_img = nib.Nifti1Image(mask_cb, nii.affine)
        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    else:
        raise ValueError("The nifti file does not have 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a rectangle mask from the input file. "
                       f"The nifti file is converted to a numpy array. If this array is in 2 dimensions, then a binary"
                       f" mask is created from this array in the form of a rectangle of lengths defined in 'size'. This"
                       f" rectangle is centered according to the 2 dimensions indicated in 'center'. If this array is "
                       f"in 3 dimensions, a binary mask is created in the shape of rectangle for each slice of the 3rd"
                       f" dimension of the array, in the same way as for a 2D array. The masks of all these slices are "
                       f"grouped in an array to form a binary mask in 3 dimensions. The mask is stored by default under"
                       f" the name 'mask.nii.gz' in the output folder."
                       f"Return an output nifti file with square mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. This nifti file must have 2D or 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.curdir,
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-size', nargs=2, required=True, type=int,
              help="(int): Length of the side of the box along first and second dimension (in pixels). (nargs=2)")
@click.option('-center', nargs=2, type=int, default=(None, None),
              help="(int): Center of the box along first and second dimension (in pixels). If no center is "
                   "provided (None), the middle is used. (nargs=2) (default: None, None)")
def rect(fname_input, output, size, center):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    if len(data.shape) == 2:
        mask_sqr = shape_square(data, size[0], size[1], center[0], center[1])  # creation of the rectangle mask
        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    elif len(data.shape) == 3:
        mask_sqr = np.zeros_like(data)  # initialization of 3D array of zeros
        for z in range(data.shape[2]):
            img_2d = data[:, :, z]  # extraction of a MRI slice (2D)
            mask_slice = shape_square(img_2d, size[0], size[1], center[0], center[1])  # creation of the mask
            # on each slice (2D)
            mask_sqr[:, :, z] = mask_slice  # addition of each masked slice to form a 3D array

        mask_sqr = mask_sqr.astype(int)
        nii_img = nib.Nifti1Image(mask_sqr, nii.affine)
        nib.save(nii_img, output)
        click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
        return output

    else:
        raise ValueError("The nifti file does not have 2 or 3 dimensions.")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help=f"Create a threshold mask from the input file. "
                       f"The nifti file is converted into a numpy array. A binary mask is created from the thresholding"
                       f" of the array. The mask is stored by default under the name 'mask.nii.gz' in the output "
                       f"folder. Return an output nifti file with threshold mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input path of the nifti file to mask. Supported extensions are .nii or .nii.gz.")
@click.option('-output', type=click.Path(), default=os.curdir,
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('-thr', default=30, help="(int): Value to threshold the data: voxels will be set to zero if their "
                                       "value is equal or less than this threshold. (default: 30)")
def threshold(fname_input, output, thr):
    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, output)
    click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
    return output


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Creates a SCT (SpinalCordToolbox) mask from the input file. Depending on the shape (cylinder,"
                       " box or Gaussian), a mask is created along z direction. To generate this mask, its center"
                       " must be specified by the user according the method."
                       " Return an output nifti file with SCT mask.")
@click.option('-input', 'fname_input', type=click.Path(), required=True,
              help="(str): Input nifti file to mask. Must be 3D. Supported extensions are .nii or .nii.gz. Example: "
                   "data.nii.gz")
@click.option('-output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              help="(str): Name of output mask. Supported extensions are .nii or .nii.gz. Example: data.nii. (default:"
                   " (os.curdir, 'mask.nii.gz')).")
@click.option('-size', default='20',
              help="(str): Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter "
                   "(Example: 35mm). If shape=gaussian, size corresponds to sigma (Example: 45). (default: 41)")
@click.option('-shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="(str): Shape of the mask. (default: cylinder)")
@click.option('-contrast', type=click.Choice(['t1', 't2', 't2s', 'dwi']), default='t2s',
              help="(str): Type of image contrast. Only with method=optic. (default: t2s)")
# @click.option('-method', type=click.Choice(['optic', 'fitseg']), default='optic',
#               help="(str): Method used for extracting the centerline: "
#                    "- optic: automatic spinal cord detection method"
#                    "- fitseg: fit a regularized centerline on an already-existing cord segmentation. It will "
#                    "interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e., every "
#                    "axial slice will exhibit a centerline pixel). (default: optic)")
# @click.option('-centerline_algo', type=click.Choice(['polyfit', 'bspline', 'linear', 'nurbs']), default='bspline',
#               help="(str): Algorithm for centerline fitting. Only relevant with -method fitseg (default: bspline)")
# @click.option('-centerline_smooth', default=30, help="(int): Degree of smoothing for centerline fitting. Only for "
#                                                      "-centerline-algo {bspline, linear}. (default: 30)")
@click.option('-centerline', type=click.Choice(['svm', 'cnn', 'viewer', 'file']), default='svm',
              help="(str): Algorithm for centerline fitting.")
@click.option('-file_centerline', type=str,
              help="(str):  Input centerline file (to use with flag -centerline file). "
                   "Example: t2_centerline_manual.nii.gz")
@click.option('-thr', type=float,
              help="Binarization threshold (between 0 and 1) to apply to the segmentation prediction. Set to -1 for no "
                   "binarization (i.e. soft segmentation output). The default threshold is specific to each contrast "
                   "and wasestimated using an optimization algorithm. More details at: "
                   "https://github.com/sct-pipeline/deepseg-threshold.")
@click.option('-brain', type=click.IntRange(0, 1),
              help="(str):  Indicate if the input image contains brain sections (to speed up segmentation). "
                   "Only use with '-centerline cnn'.")
@click.option('-kernel', type=click.Choice(['2d', '3d']), default='2d',
              help="(str):  Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with "
                   "2D kernels. (default: 2d)")
@click.option('-remove', type=click.IntRange(0, 1), default=1, help="(int): Remove temporary files. (default: 1)")
@click.option('-verbose', type=click.IntRange(0, 2), default=1,
              help="(int): Verbose: 0 = nothing, 1 = classic, 2 = expended. (default: 1)")
def sct(fname_input, fname_output, contrast, centerline, file_centerline, thr, brain, kernel, size, shape, remove, verbose):

    # Make sure input path exists
    if not os.path.exists(fname_input):
        raise RuntimeError("Input file does not exist")

    # Get the number of dimensions
    nii_input = nib.load(fname_input)
    ndim = nii_input.ndim
    # If 4d, last dimension is time, average last dim for better SNT
    if ndim == 4:
        input_3d = np.mean(nii_input.get_fdata(), 3)
        nii_3d = nib.Nifti1Image(input_3d, affine=nii_input.affine, header=nii_input.header)
        fname_mean = os.path.join(os.path.dirname(fname_output), 'mean_3d.nii.gz')
        nib.save(nii_3d, fname_mean)
        fname_process = fname_mean
    # If not then only set the processing filename
    else:
        fname_process = fname_input

    # # Get the centerline
    # method = 'fitseg'
    # centerline_algo = 'linear'
    # centerline_smooth = 30
    # path_centerline = os.path.join(os.path.dirname(fname_output), 'centerline')
    # fname_seg = path_centerline + '.nii.gz'
    # if method == "optic":
    #     run_subprocess(f"sct_get_centerline -i {fname_process} -c {contrast} -o {path_centerline} -v {str(verbose)}")
    #
    # elif method == "fitseg" and (centerline_algo == "polyfit" or centerline_algo == "nurbs"):
    #     run_subprocess(f"sct_get_centerline -i {fname_process} -method {method} -centerline-algo {centerline_algo} "
    #                    f"-o {path_centerline} -v {str(verbose)}")
    #
    # elif method == "fitseg" and (centerline_algo == "bspline" or centerline_algo == "linear"):
    #     run_subprocess(f"sct_get_centerline -i {fname_process} -method {method} -centerline-algo {centerline_algo} "
    #                    f"-centerline-smooth {str(centerline_smooth)} -o {path_centerline} -v {str(verbose)}")
    #
    # else:
    #     raise ValueError("Could not get centerline.")

    # Run sct_deepseg_sc
    fname_seg = os.path.join(os.path.dirname(fname_output), 'seg.nii.gz')
    cmd = f"sct_deepseg_sc -i {fname_process} -o {fname_seg} -c {contrast} -centerline {centerline} -kernel {kernel} " \
          f"-r {remove}"
    if centerline == 'file':
        cmd += f" -file_centerline {file_centerline}"
    if thr is not None:
        cmd += f" -thr {thr}"
    if brain is not None and centerline == 'cnn':
        cmd += f" -brain {brain}"

    run_subprocess(cmd)

    # Create the mask
    run_subprocess(f"sct_create_mask -i {fname_process} -p centerline,{fname_seg} -size {size} -f {shape} "
                   f"-o {fname_output} -r {str(remove)} -v {str(verbose)}")

    if remove:
        os.remove(fname_seg)
        if ndim == 4:
            os.remove(fname_mean)

    click.echo(f"The path for the output mask is: {os.path.abspath(fname_output)}")
    return fname_output
