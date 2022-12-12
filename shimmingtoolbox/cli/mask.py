#!/usr/bin/env python3

import click
import nibabel as nib
import numpy as np
import os
import logging

import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.shapes import shape_square
from shimmingtoolbox.masking.shapes import shape_cube
from shimmingtoolbox.utils import run_subprocess, create_output_dir, set_all_loggers

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Create a mask based on a specified shape (box, rectangle, SpinalCord Toolbox mask) or based on the "
                  "thresholding of an input image. Callable with the prefix 'st' in front of 'mask'. "
                  "(Example: 'st_mask -h').")
def mask_cli():
    pass


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Create a box mask from the input file. The nifti file is converted to a numpy array. If this "
                       "array is in 3D dimensions, then a binary mask is created from this array in the form of a box "
                       "with lengths defined in 'size'. This box is centered according to the 3 dimensions indicated "
                       "in 'center'. The mask is stored by default under the name 'mask.nii.gz' in the output folder."
                       "Return the filename for the output mask.")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask. This nifti file must have 3D. Supported extensions are"
                   " .nii or .nii.gz.")
@click.option('-o', '--output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'), show_default=True,
              help="Name of output mask. Supported extensions are .nii or .nii.gz.")
@click.option('--size', nargs=3, required=True, type=int,
              help="Length of the side of the box along first, second and third dimension (in pixels). "
                   "(nargs=3)")
@click.option('--center', nargs=3, type=int, default=(None, None, None),
              help="Center of the box along first, second and third dimension (in pixels). If no center "
                   "is provided (None), the middle is used. (nargs=3) (default: None, None, None)")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def box(fname_input, output, size, center, verbose):

    # Set all loggers
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(output, is_file=True)

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
                  help="Create a rectangle mask from the input file. "
                       "The nifti file is converted to a numpy array. If this array is in 2 dimensions, then a binary"
                       " mask is created from this array in the form of a rectangle of lengths defined in 'size'. This"
                       " rectangle is centered according to the 2 dimensions indicated in 'center'. If this array is "
                       "in 3 dimensions, a binary mask is created in the shape of rectangle for each slice of the 3rd"
                       " dimension of the array, in the same way as for a 2D array. The masks of all these slices are "
                       "grouped in an array to form a binary mask in 3 dimensions. The mask is stored by default under"
                       " the name 'mask.nii.gz' in the output folder."
                       "Return an output nifti file with square mask.")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask. This nifti file must have 2D or 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-o', '--output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'), show_default=True,
              help="Name of output mask. Supported extensions are .nii or .nii.gz.")
@click.option('--size', nargs=2, required=True, type=int,
              help="Length of the side of the box along first and second dimension (in pixels). (nargs=2)")
@click.option('--center', nargs=2, type=int, default=(None, None),
              help="Center of the box along first and second dimension (in pixels). If no center is "
                   "provided (None), the middle is used. (nargs=2) (default: None, None)")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def rect(fname_input, output, size, center, verbose):

    # Set all loggers
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(output, is_file=True)

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
                  help="Create a spherical mask in the coordinates of the input file. The mask is stored by default "
                       "under the name 'mask.nii.gz' in the output folder.")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask. This nifti file must be 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              show_default=True, help="Name of output mask. Supported extensions are .nii or .nii.gz.")
@click.option('-r', '--radius', required=True, type=int,
              help="Number of pixels for the radius of the sphere.")
@click.option('--center', nargs=3, type=int, default=(None, None, None),
              help="Center of the sphere along first, second and third dimension (in pixels). If no center is "
                   "provided, the middle is used.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def sphere(fname_input, fname_output, radius, center, verbose):

    # Set all loggers
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    nii = nib.load(fname_input)
    shape = nii.shape

    # Defaults to the center of the array if no center is provided
    if center[0] is None:
        center[0] = shape[0] // 2
    if center[1] is None:
        center[1] = shape[1] // 2
    if center[2] is None:
        center[2] = shape[2] // 2

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    mask = dist <= radius

    nii_mask = nib.Nifti1Image(mask.astype(int), affine=nii.affine, header=nii.header)
    logger.info(fname_output)
    nib.save(nii_mask, fname_output)


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Create a threshold mask from the input file. "
                       "The nifti file is converted into a numpy array. A binary mask is created from the thresholding"
                       " of the array. The mask is stored by default under the name 'mask.nii.gz' in the output "
                       "folder. Return an output nifti file with threshold mask.")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask. Supported extensions are .nii or .nii.gz.")
@click.option('-o', '--output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'), show_default=True,
              help="Name of output mask. Supported extensions are .nii or .nii.gz. (default: "
                   "(os.curdir, 'mask.nii.gz'))")
@click.option('--thr', default=30, help="Value to threshold the data: voxels will be set to zero if their "
                                        "value is equal or less than this threshold. (default: 30)")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def threshold(fname_input, output, thr):
    # Prepare the output
    create_output_dir(output, is_file=True)

    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(int)
    nii_img = nib.Nifti1Image(mask_thr, nii.affine)
    nib.save(nii_img, output)
    click.echo(f"The filename for the output mask is: {os.path.abspath(output)}")
    return output


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="""Creates a mask around the spinal cord using the Spinal Cord Toolbox (SCT). The mask, which
                   size can be specified, requires to identify the spinal cord centerline. The method of identification
                   is specified by the flag '--centerline'. The output of this function is a NIfTI file containing the
                   mask.""")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input nifti file to mask. Must be 3D. Supported extensions are .nii or .nii.gz. Example: "
                   "data.nii.gz")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              show_default=True,
              help="Name of output mask. Supported extensions are .nii or .nii.gz. Example: data.nii.")
@click.option('--size', default='20', type=int, show_default=True,
              help="Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter "
                   "(Example: 35mm). If shape=gaussian, size corresponds to sigma (Example: 45).")
@click.option('--shape', type=click.Choice(['cylinder', 'box', 'gaussian']), default='cylinder',
              help="Shape of the mask.")
@click.option('--contrast', type=click.Choice(['t1', 't2', 't2s', 'dwi']), default='t2s', show_default=True,
              help="Type of image contrast.")
@click.option('--centerline', type=click.Choice(['svm', 'cnn', 'viewer', 'file']), default='svm', show_default=True,
              help="""
              Method used for extracting the centerline:
              - svm: Automatic detection using Support Vector Machine algorithm.
              - cnn: Automatic detection using Convolutional Neural Network.
              - viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer
              followed by regularization.
              - file: Use an existing centerline
              (use with flag --file_centerline)""")
@click.option('--file-centerline', 'file_centerline', type=click.Path(),
              help="Input centerline file. This option is only valid with '--centerline file'. "
                   "Example: t2_centerline_manual.nii.gz")
@click.option('--brain', type=click.IntRange(0, 1),
              help="Set to 1 if the image contains the brain (or part of it), set to 0 otherwise "
                   "(to speed up the segmentation). This option is only valid with '--centerline cnn'.")
@click.option('--kernel', type=click.Choice(['2d', '3d']), default='2d', show_default=True,
              help="Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with "
                   "2D kernels.")
@click.option('--remove-tmp', 'remove_tmp', type=bool, default=True, show_default=True,
              help="Remove temporary files.")
@click.option('--verbose', type=click.IntRange(0, 2), default=1, show_default=True,
              help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
# Options for _get_centerline
# @click.option('--method', type=click.Choice(['optic', 'fitseg']), default='optic',
#               help="(str): Method used for extracting the centerline: "
#                    "- optic: automatic spinal cord detection method"
#                    "- fitseg: fit a regularized centerline on an already-existing cord segmentation. It will "
#                    "interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e., "
#                    "every axial slice will exhibit a centerline pixel). (default: optic)")
# @click.option('--centerline_algo', type=click.Choice(['polyfit', 'bspline', 'linear', 'nurbs']), default='bspline',
#               help="(str): Algorithm for centerline fitting. Only relevant with -method fitseg (default: bspline)")
# @click.option('--centerline_smooth', default=30, help="(int): Degree of smoothing for centerline fitting. Only for "
#                                                      "-centerline-algo {bspline, linear}. (default: 30)")
def sct(fname_input, fname_output, contrast, centerline, file_centerline, brain, kernel, size, shape, remove_tmp,
        verbose):

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Make sure input path exists
    if not os.path.exists(fname_input):
        raise RuntimeError("Input file does not exist")

    # Get the number of dimensions
    nii_input = nib.load(fname_input)
    ndim = nii_input.ndim
    # If 4d, last dimension is time, average last dim for better SNR
    if ndim == 4:
        input_3d = np.mean(nii_input.get_fdata(), 3)
        nii_3d = nib.Nifti1Image(input_3d, affine=nii_input.affine, header=nii_input.header)
        fname_mean = os.path.join(os.path.dirname(fname_output), 'mean_3d.nii.gz')
        nib.save(nii_3d, fname_mean)
        fname_process = fname_mean
    # If not then only set the processing filename
    else:
        fname_process = fname_input

    fname_seg = os.path.join(os.path.dirname(fname_output), 'seg.nii.gz')

    # sct_get_centerline is faster than sct_deepseg_sc, however, it is a bit less accurate. More investigations needed
    # in the future, this code is commented out so that we can persue investigation.
    # # Get the centerline
    # _get_centerline(fname_process, fname_seg)

    # Run sct_deepseg_sc
    # Use sct parameter convention
    if remove_tmp:
        remove = 1
    else:
        remove = 0

    cmd = ['sct_deepseg_sc', '-i', fname_process, '-o', fname_seg, '-c', contrast, '-centerline', centerline,
           '-kernel', kernel, '-r', str(remove), '-v', str(verbose)]

    if centerline == 'file':
        cmd += ['-file_centerline', file_centerline]
    if brain is not None and centerline == 'cnn':
        cmd += ['-brain', str(brain)]

    run_subprocess(cmd)

    # Create the mask
    run_subprocess(['sct_create_mask', '-i', fname_process, '-p', f"centerline,{fname_seg}", '-size', str(size),
                    '-f', shape, '-o', fname_output, '-r', str(remove), '-v', str(verbose)])

    if remove:
        os.remove(fname_seg)
        if ndim == 4:
            os.remove(fname_mean)

    click.echo(f"The path for the output mask is: {os.path.abspath(fname_output)}")
    return fname_output


# def _get_centerline(fname_process, fname_output, method='optic', contrast='t2', centerline_algo='bspline',
#                     centerline_smooth='30', verbose='1'):
#     """ Wrapper to sct_get_centerline. Allows to get the centerline of the spinal cord and outputs a nifti file
#     containing the output mask.
#
#     Args:
#         fname_process (str): Input filename containing the spinal cord image. Supported extensions are .nii or
#                              .nii.gz.
#         fname_output (str): Output filename containing the senterline of the spinal cord.Supported extensions is
#                             ".nii.gz".
#         method (str): Method used for extracting the centerline:
#                       - optic: automatic spinal cord detection method
#                       - fitseg: fit a regularized centerline on an already-existing cord segmentation. It will
#                       interpolate if slices are missing and extrapolate beyond the segmentation boundaries
#                       (i.e., every axial slice will exhibit a centerline pixel).
#         contrast (str): Type of image contrast. Supported contrast: t1, t2, t2s, dwi.
#         centerline_algo (str): Algorithm for centerline fitting. Only relevant with -method fitseg.
#                          Supported algo: polyfit, bspline, linear, nurbs.
#         centerline_smooth (int): Degree of smoothing for centerline fitting.
#                                  Only for -centerline-algo {bspline, linear}.
#         verbose (int): Verbose: 0 = nothing, 1 = classic, 2 = expended.
#
#     Returns:
#
#     """
#     path_seg = fname_output.rsplit('.nii.gz', 1)[0]
#
#     if method == "optic":
#         run_subprocess(f"sct_get_centerline -i {fname_process} -c {contrast} -o {path_seg} -v {str(verbose)}")
#
#     elif method == "fitseg" and (centerline_algo == "polyfit" or centerline_algo == "nurbs"):
#         run_subprocess(f"sct_get_centerline -i {fname_process} -method {method} -centerline-algo {centerline_algo} "
#                        f"-o {path_seg} -v {str(verbose)}")
#
#     elif method == "fitseg" and (centerline_algo == "bspline" or centerline_algo == "linear"):
#         run_subprocess(f"sct_get_centerline -i {fname_process} -method {method} -centerline-algo {centerline_algo} "
#                        f"-centerline-smooth {str(centerline_smooth)} -o {path_seg} -v {str(verbose)}")
#
#     else:
#         raise ValueError("Could not get centerline.")
