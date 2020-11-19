#!/usr/bin/env python3

import click
import os
import subprocess

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, help=f"Perform dilatation, erosion, deep segmentation and get "
                                                     f"centerline of images.")
def sct_features():
    pass


@sct_features.command(context_settings=CONTEXT_SETTINGS, help=f"Perform mathematical dilatations on images")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output folder for mask in nifti file")
@click.option('-dilate', type=int, required=True,
              help="Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds"
                   "to the length of an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the"
                   " radius, not including the center element (size=0 has no effect).")
@click.option('-shape', type=click.Choice(['square', 'cube', 'disk', 'ball']), default='ball',
              help="Shape of the structuring element for the mathematical morphology operation. Default: ball. If a 2D "
                   "shape {'disk', 'square'} is selected, -dim must be specified. (default: ball)")
@click.option('-dim', type=click.IntRange(0, 2),
              help="Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish "
                   "to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, "
                   "dim=2.")
@click.option('-symmetrize', type=click.IntRange(0, 2), default=1, help="Symmetrize data along the specified "
                                                                        "dimension.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
def erode(fname_input, output, dilate, shape, dim, symmetrize, verbose):
    """
            Perform mathematical dilatations on images. Some inputs can be either a number or a 4d image or several 3d
            images separated with ","

            Args:
                fname_input (str): Input image. Example: t1.nii.gz
                output (str): Output folder for cube mask.
                dilate (int): Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size
                        corresponds to the length of an edge (size=1 has no effect). If shape={'disk', 'ball'}: size
                        corresponds to the radius, not including the center element (size=0 has no effect).
                shape (str): Shape of the structuring element for the mathematical morphology operation. Default: ball.
                        If a 2D shape {'disk', 'square'} is selected, -dim must be specified. (default: ball)
                dim (int): Dimension of the array which 2D structural element will be orthogonal to. For example, if
                        you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be:
                        shape=disk, dim=2.
                symmetrize (int): Symmetrize data along the specified dimension.
                verbose (int): Verbose. 0: nothing. 1: basic. 2: extended. (default: 1)

            Return:
                output (str): Filename for the output mask.
            """
    subprocess.run(['bash', '-c', 'sct_maths', '-i', fname_input, '-o', output, '-dilate', dilate, '-shape', shape,
                    '-dim', dim, '-symmetrize', symmetrize, '-verbose', verbose])
    click.echo(f"The path for the output mask is: {os.path.abspath(output)}")
    return output


@sct_features.command(context_settings=CONTEXT_SETTINGS, help=f"Perform mathematical erosions on images")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file")
@click.option('-output', type=click.Path(), default=os.curdir, help="Output folder for mask in nifti file")
@click.option('-erode', type=int, required=True,
              help="Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds"
                   "to the length of an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the"
                   " radius, not including the center element (size=0 has no effect).")
@click.option('-shape', type=click.Choice(['square', 'cube', 'disk', 'ball']), default='ball',
              help="Shape of the structuring element for the mathematical morphology operation. Default: ball. If a 2D "
                   "shape {'disk', 'square'} is selected, -dim must be specified. (default: ball)")
@click.option('-dim', type=click.IntRange(0, 2),
              help="Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish "
                   "to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, "
                   "dim=2.")
@click.option('-symmetrize', type=click.IntRange(0, 2), default=1, help="Symmetrize data along the specified "
                                                                        "dimension.")
@click.option('-verbose', type=click.IntRange(0, 2), default=1, help="Verbose: 0 = nothing, 1 = classic, 2 = expended.")
def erode(fname_input, output, erode, shape, dim, symmetrize, verbose):
    """
            Perform mathematical erosions on images. Some inputs can be either a number or a 4d image or several 3d
            images separated with ","

            Args:
                fname_input (str): Input image. Example: t1.nii.gz
                output (str): Output folder for cube mask.
                erode (int): Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size
                        corresponds to the length of an edge (size=1 has no effect). If shape={'disk', 'ball'}: size
                        corresponds to the radius, not including the center element (size=0 has no effect).
                shape (str): Shape of the structuring element for the mathematical morphology operation. Default: ball.
                        If a 2D shape {'disk', 'square'} is selected, -dim must be specified. (default: ball)
                dim (int): Dimension of the array which 2D structural element will be orthogonal to. For example, if
                        you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be:
                        shape=disk, dim=2.
                symmetrize (int): Symmetrize data along the specified dimension.
                verbose (int): Verbose. 0: nothing. 1: basic. 2: extended. (default: 1)

            Return:
                output (str): Filename for the output mask.
            """
    subprocess.run(['bash', '-c', 'sct_maths', '-i', fname_input, '-o', output, '-erode', erode, '-shape', shape,
                    '-dim', dim, '-symmetrize', symmetrize, '-verbose', verbose])
    click.echo(f"The path for the output mask is: {os.path.abspath(output)}")
    return output


@sct_features.command(context_settings=CONTEXT_SETTINGS,
                      help=f"Spinal Cord Segmentation using convolutional networks. Reference: Gros et al. Automatic "
                           f"segmentation of the spinal cord and intramedullary multiple sclerosis lesions with "
                           f"convolutional neural networks.")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file")
@click.option('-c', type=click.Choice(['t1', 't2', 't2s', 'dwi']), required=True, help="Type of image contrast.")
@click.option('-centerline', type=click.Choice(['svm', 'cnn', 'viewer', 'file']), default='svm',
              help="Method used for extracting the centerline: "
                   "svm: Automatic detection using Support Vector Machine algorithm."
                   "cnn: Automatic detection using Convolutional Neural Network."
                   "viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer "
                   "followed by regularization. "
                   "file: Use an existing centerline (use with flag -file_centerline) (default: svm)")
@click.option('-file_centerline', type=click.Path(), help="Input centerline file (to use with flag -centerline file). "
                                                          "Example: t2_centerline_manual.nii.gz")
@click.option('-thr', type=float, required=True,
              help="Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds"
                   "to the length of an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the"
                   " radius, not including the center element (size=0 has no effect).")
@click.option('-brain', type=click.IntRange(0, 1), help="Indicate if the input image contains brain sections (to speed "
                                                        "up segmentation). Only use with '-centerline cnn'.")
@click.option('-kernel', type=click.Choice(['2d', '3d']), default='2d',
              help="Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with 2D kernels. "
                   "(default: 2d)")
@click.option('-ofolder', type=click.Path(), default='/home/docs/checkouts/readthedocs.org/user_builds/'
                                                     'spinalcordtoolbox/checkouts/latest/documentation/source)',
              help="Output folder. Example: My_Output_Folder/ (default: /home/docs/checkouts/readthedocs.org/"
                   "user_builds/spinalcordtoolbox/checkouts/latest/documentation/source)")
@click.option('-r', type=click.IntRange(0, 1), default=1, help="Remove temporary files. (default: 1)")
@click.option('-v', type=click.IntRange(0, 2), default=1, help="1: display on (default), 0: display off, 2: extended "
                                                               "(default: 1)")
@click.option('-qc', help="The path where the quality control generated content will be saved")
@click.option('-qc_dataset', help="If provided, this string will be mentioned in the QC report as the dataset the "
                                  "process was run on")
@click.option('-qc_subject', help="If provided, this string will be mentioned in the QC report as the subject the "
                                  "process was run on")
@click.option('-igt', help="File name of ground-truth segmentation.")
def deepseg_sc(fname_input, c, centerline, file_centerline, thr, brain, kernel, ofolder, r, v, qc, qc_dataset, qc_subject, igt):
    """
            Perform mathematical erosions on images. Some inputs can be either a number or a 4d image or several 3d
            images separated with ","

            Args:
                fname_input (str): Input image. Example: t1.nii.gz
                c (str): Type of image contrast.
                centerline (str): Method used for extracting the centerline:
                         svm: Automatic detection using Support Vector Machine algorithm.
                         cnn: Automatic detection using Convolutional Neural Network.
                         viewer: Semi-automatic detection using manual selection of a few points with an interactive
                         viewer followed by regularization.
                         file: Use an existing centerline (use with flag -file_centerline) (default: svm)
                file_centerline (str): Input centerline file (to use with flag -centerline file).
                        Example: t2_centerline_manual.nii.gz
                thr (float): Binarization threshold (between 0 and 1) to apply to the segmentation prediction. Set to
                        -1 for no binarization (i.e. soft segmentation output). The default threshold is specific to
                        each contrast and wasestimated using an optimization algorithm. More details at:
                        https://github.com/sct-pipeline/deepseg-threshold.
                brain (int): Indicate if the input image contains brain sections (to speed up segmentation). Only use
                        with "-centerline cnn".
                kernel (str): Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with 2D
                        kernels. (default: 2d)
                ofolder (str): Output folder for cube mask.
                r (int): Remove temporary files. (default: 1)
                v (int): 1: display on (default), 0: display off, 2: extended (default: 1)
                qc (str): The path where the quality control generated content will be saved
                qc_dataset (str): If provided, this string will be mentioned in the QC report as the dataset the process
                        was run on
                qc_subject (str): If provided, this string will be mentioned in the QC report as the subject the process
                        was run on
                igt (str): File name of ground-truth segmentation.

            Return:
                output (str): Filename for the output mask.
            """
    subprocess.run(['bash', '-c', 'sct_deepseg_sc', '-i', fname_input, '-c', c, '-centerline', centerline, '-file_centerline', file_centerline,
                    '-thr', thr, '-brain', brain, '-kernel', kernel, '-ofolder', ofolder, '-r', r, '-v', v, '-qc', qc, '-qc-dataset', qc_dataset
                    , '-qc_subject', qc_subject, '-igt', igt])
    click.echo(f"The path for the output mask is: {os.path.abspath(ofolder)}")
    return ofolder
