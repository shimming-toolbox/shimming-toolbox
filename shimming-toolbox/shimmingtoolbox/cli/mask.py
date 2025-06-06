#!/usr/bin/env python3

import pathlib
import click
import logging
import nibabel as nib
import numpy as np
import os

from shimmingtoolbox.masking.shapes import shape_square, shape_cube, shape_sphere
import shimmingtoolbox.masking.threshold
from shimmingtoolbox.masking.mask_mrs import mask_mrs
from shimmingtoolbox.utils import run_subprocess, create_output_dir, set_all_loggers
from shimmingtoolbox.masking.mask_utils import modify_binary_mask as modify_binary_mask_api
from shimmingtoolbox.masking.mask_utils import SOFTMASK_FUNCS, save_softmask

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
        mask_cb = mask_cb.astype(np.int32)
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

        mask_sqr = mask_sqr.astype(np.int32)
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

    # API for sphere mask
    mask = shape_sphere(nii.get_fdata(), radius, center[0], center[1], center[2])

    nii_mask = nib.Nifti1Image(mask.astype(np.int32), affine=nii.affine, header=nii.header)
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
@click.option('--thr', type=click.FLOAT, default=30.0, show_default=True,
              help="Value to threshold the data: voxels will be set to zero if their value is equal or less than this"
                   " threshold.")
@click.option('--scaled-thr', is_flag=True, default=False, show_default=True,
              help="Indicate whether the --thr option is scaled from 0 to 1 (True) or according to the values within "
                   "the input (False). Default: False")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def threshold(fname_input, output, thr, scaled_thr, verbose):

    # Set all loggers
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(output, is_file=True)

    nii = nib.load(fname_input)
    data = nii.get_fdata()  # convert nifti file to numpy array

    mask_thr = shimmingtoolbox.masking.threshold.threshold(data, thr, scaled_thr)  # creation of the threshold mask
    mask_thr = mask_thr.astype(np.int32)
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


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Wrapper for BET, please see https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/bet. "
                    "Create a brain mask in the coordinates of the input file. The mask is stored by default "
                       "under the name 'mask.nii.gz' in the output folder.")
@click.option('-i', '--input', 'fname_input', type=click.Path(exists=True), required=True,
              help="Input path of the nifti file to mask. This nifti file must be 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'mask'),
              show_default=True, help="Name of output mask. Do not add extension")
@click.option('-f', '--f_param', required=False, type=float, default=0.5,
              help="fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates")
@click.option('-g', '--g_param', type=float, required=False, default=0,
              help="vertical gradient in fractional intensity threshold (-1->1); positive values give larger brain outline at bottom, smaller at top")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def bet(fname_input, fname_output, f_param, g_param, verbose):

    set_all_loggers(verbose)
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

    # Run BET
    # Create the mask
    run_subprocess(['bet2', fname_process, fname_output, '-f', str(f_param), '-g', str(g_param), '-m'])

    return fname_output


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Wrapper for modify_binary_mask. Lets the user dilate or erode their masks")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the nifti file to mask. This nifti file must be 3D. Supported "
                   "extensions are .nii or .nii.gz.")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'mask.nii.gz'),
              show_default=True, help="Name of output mask. Supported extensions are .nii or .nii.gz.")
@click.option('--shape', 'shape', required=False, type=click.Choice(['sphere', 'cross', 'line', 'cube', 'None']), default='sphere',
              help="3d kernel to perform the dilation. Allowed shapes are: 'sphere', 'cross', 'line', 'cube', 'None'.")
@click.option('--size', 'size', type=float, required=True, default=1,
              help="Kernel size for the dilation or erosion. Must be odd.")
@click.option('--operation', 'operation', type=click.Choice(['erode', 'dilate']), required=True, default="dilate",
              help="operation to perform. Allowed operations are: 'dilate', 'erode'.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def modify_binary_mask(fname_input, fname_output, shape, size, operation, verbose):

    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Make sure input path exists
    if not os.path.exists(fname_input):
        raise RuntimeError("Input file does not exist")

    # Run modify_binary_mask
    nii = nib.load(fname_input)
    array = nii.get_fdata()
    mask = modify_binary_mask_api(array, shape, size, operation)

    nii_mask = nib.Nifti1Image(mask, affine=nii.affine, header=nii.header)
    nib.save(nii_mask, fname_output)

    # Look for a json file with the same name as the input file
    path = pathlib.Path(fname_input)
    while path.suffix:
        path = path.with_suffix('')
    fname_json = str(path.with_suffix('.json'))
    click.echo(f"Looking for json file at {fname_json}")
    if os.path.exists(fname_json):
        path = pathlib.Path(fname_output)
        while path.suffix:
            path = path.with_suffix('')
        fname_output_json = str(path.with_suffix('.json'))
        click.echo(f"Found json file at {fname_json}")
        with open(fname_json, 'r') as f:
            json_data = f.read()
        with open(fname_output_json, 'w') as f:
            click.echo(f"Copying json file from {fname_json} to {fname_output_json}")
            f.write(json_data)

    return fname_output


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Create a mask to shim single voxel MRS. "
                       "Voxel position and size can be directly given or these info can be read "
                       "from the siemens raw-data. "
                       "The mask is stored by default under the name 'mask_mrs.nii.gz' in the output "
                       "folder. Return an output nifti file to be used as a mask for MRS shimming.")
@click.option('-i', '--input', 'fname_input', type=click.Path(), required=True,
              help="Input path of the fieldmap to be shimmed.")
@click.option('-r', '--raw', 'raw_data', type=click.Path(),
              help="Input path of the raw-data (supported extention .rda)")
@click.option('-o', '--output', type=click.Path(), default=os.path.join(os.curdir, 'mask_mrs.nii.gz'),
              show_default=True, help="Name of the output mask. Supported extensions are .nii or .nii.gz.")
@click.option('-c', '--center', nargs=3, type=click.FLOAT, help="Voxel's center position in mm of the x, y and z of "
              "the scanner's coordinate")
@click.option('-s', '--size', nargs=3, type=click.FLOAT, help="Voxel size in mm of the x, y and z of the scanner's "
              "coordinate")
@click.option('--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def mrs(fname_input, output, raw_data, center, size, verbose):

    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(output, is_file=True)

    nii = nib.load(fname_input)
    output_mask = mask_mrs(fname_input, raw_data, center, size)  # creation of the MRS mask
    output_mask = output_mask.astype(np.int32)
    nii_img = nib.Nifti1Image(output_mask, nii.affine, header=nii.header)
    nib.save(nii_img, output)
    logger.info(f"The filename for the output mask is: {os.path.abspath(output)}")


@mask_cli.command(context_settings=CONTEXT_SETTINGS,
                  help="Creates a soft mask by creating a blur zone around the binary mask.")
@click.option('-i', '--input', 'fname_input_binmask', type=click.Path(), required=True,
              help="Path to the binary mask. Supported extensions are .nii or .nii.gz.")
@click.option('-is', '--input-softmask', 'fname_input_softmask', type=click.Path(), default=None,
              help="Path to an existing soft mask. Use only on sum-type softmask. Supported extensions are .nii or .nii.gz.")
@click.option('-o', '--output', 'fname_output_softmask', type=click.Path(), default=os.path.join(os.curdir, 'softmask.nii.gz'),
              show_default=True, help="Path to the output soft mask. Supported extensions are .nii or .nii.gz.")
@click.option('-t', '--type', type=click.Choice(list(SOFTMASK_FUNCS.keys())), default='2levels',
              help=f"Type of soft mask. Allowed: {', '.join(SOFTMASK_FUNCS.keys())}")
@click.option('-bw', '--blur-width', 'blur_width', default = '6mm',
              help="Width of the blurred zone.")
@click.option('-bu', '--blur-units', 'blur_units', type=click.Choice(['mm', 'px']), default='mm',
              help="Units of the blur width. Can be in pixels (px) or in millimeters (mm).")
@click.option('-bv', '--blur-value', 'blur_value', default = 0.5,
              help="Intensity of the coefficients in the blurred zone. Use only on 2levels-type softmask")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def create_softmask(fname_input_binmask, fname_input_softmask, fname_output_softmask, type, blur_width, blur_units, blur_value, verbose) :

    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output_softmask, is_file=True)

    output_softmask = create_softmask(fname_input_binmask, fname_input_softmask, type, blur_width, blur_units, blur_value)
    save_softmask(output_softmask, fname_output_softmask, fname_input_binmask)
