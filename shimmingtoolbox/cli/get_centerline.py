#!/usr/bin/env python3

import click
import os
import subprocess

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS, help="Extract the spinal cord centerline. Necessary if you want to "
                                                       "apply a centerline process mask on your spinal cord MRI.")
@click.option('-input', 'fname_input', type=click.Path(), required=True, help="Input path of the nifti file")
@click.option('-c', type=click.Choice(['t1', 't2', 't2s', 'dwi']),
              help="Type of image contrast. Only with method=optic.")
@click.option('-method', type=click.Choice(['optic', 'viewer', 'fitseg']), default='optic',
              help="Method used for extracting the centerline: "
                   "- optic: automatic spinal cord detection method"
                   "- viewer: manual selection a few points followed by interpolation"
                   "- fitseg: fit a regularized centerline on an already-existing cord segmentation. It will "
                   "interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e., every "
                   "axial slice will exhibit a centerline pixel). (default: optic)")
@click.option('-centerline_algo', type=click.Choice(['polyfit', 'bspline', 'linear', 'nurbs']), default='bspline',
              help="Algorithm for centerline fitting. Only relevant with -method fitseg (default: bspline)")
@click.option('-centerline_smooth', default=30, help="Degree of smoothing for centerline fitting. Only for "
                                                     "-centerline-algo {bspline, linear}. (default: 30)")
@click.option('-output', type=click.Path(),
              help="File name (without extension) for the centerline output files. By default, output file will be the "
                   "input with suffix '_centerline'. Example: 'centerline_optic'")
@click.option('-gap', default=20.0, help="Gap in mm between manually selected points. Only with method=viewer. "
                                         "(default: 20.0)")
@click.option('-v', type=click.IntRange(0, 2), default=1, help="1: display on (default), 0: display off, 2: extended "
                                                               "(default: 1)")
def get_centerline_cli(fname_input, c, method, centerline_algo, centerline_smooth, output, gap, v):
    """
            This function extracts the spinal cord centerline. Three methods are available: 'optic' (automatic),
            'viewer' (manual), and 'fitseg' (applied on segmented image). These functions output (i) a NIFTI file with
            labels corresponding to the discrete centerline, and (ii) a csv file containing the float (more precise)
            coordinates of the centerline in the RPI orientation.

            Args:
                fname_input (str): Input image. Example: t1.nii.gz
                c (str): Type of image contrast. Only with method=optic.
                method (str): Method used for extracting the centerline.
                          - optic: automatic spinal cord detection method
                          - viewer: manual selection a few points followed by interpolation
                          - fitseg: fit a regularized centerline on an already-existing cord segmentation. It will
                          interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e.,
                          every axial slice will exhibit a centerline pixel). (default: optic)
                centerline_algo (str): Algorithm for centerline fitting. Only relevant with -method fitseg
                        (default: bspline)
                centerline_smooth (int): Degree of smoothing for centerline fitting. Only for -centerline-algo {bspline,
                        linear}. (default: 30)
                output (str): Output folder for cube mask.
                gap (float): Gap in mm between manually selected points. Only with method=viewer. (default: 20.0)
                v (int): 1: display on (default), 0: display off, 2: extended (default: 1)

            Return:
                output (str): Filename for the output mask.
            """
    # Create the folder where the nifti file will be stored
    if not os.path.exists(output):
        os.makedirs(output)

    if method == "optic":
        subprocess.run(['sct_get_centerline', '-i', fname_input, '-c', c, '-o', output, '-v', v], check=True)

    elif method == "fitseg" and (centerline_algo == "polyfit" or centerline_algo == "nurbs"):
        subprocess.run(['sct_get_centerline', '-i', fname_input, '-method', method, '-centerline-algo', centerline_algo,
                        '-o', output, '-v', v], check=True)

    elif method == "fitseg" and (centerline_algo == "bspline" or centerline_algo == "linear"):
        subprocess.run(['sct_get_centerline', '-i', fname_input, '-method', method, '-centerline-algo', centerline_algo,
                        '-centerline-smooth', centerline_smooth, '-o', output, '-v', v], check=True)

    elif method == "viewer":
        subprocess.run(['sct_get_centerline', '-i', fname_input, '-method', method, '-o', output, '-gap', gap, '-v', v],
                       check=True)

    else:
        raise ValueError("The implementation of get_centerline_cli is bad. Run get_centerline_cli -h for more "
                         "information on how to call this command.")

    click.echo(f"The path for the output mask is: {os.path.abspath(output)}")
    return output
