# -*- coding: utf-8 -*-
"""
- fm_static
- fm_rt
- gradient_rt
"""

import click
import os
import nibabel as nib
import numpy as np
import logging
import json
import math

from shimmingtoolbox.cli.realtime_shim import realtime_shim_cli
from shimmingtoolbox.shim.sequencer import shim_sequencer
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.coils.coordinates import generate_meshgrid
from shimmingtoolbox.shim.sequencer import update_affine_for_ap_slices
from shimmingtoolbox.shim.sequencer import extend_slice
from shimmingtoolbox.shim.sequencer import define_slices
from shimmingtoolbox import __dir_config_scanner_constraints__
from shimmingtoolbox.utils import create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Shim according to the specified algorithm as an argument e.g. st_shim xxxxx")
def shim_cli():
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils', nargs=2, multiple=True, type=(click.Path(exists=True), click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__dir_config_scanner_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="B0 fieldmap. This should be a 3d file.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Filename of the anatomical image to apply the correction.")
@click.option('--mask', 'fname_mask_anat', type=click.Path(exists=True), required=False,
              help="3D nifti file used to define the spatial region to shim. "
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('--scanner-coil-order', type=click.INT, default=0, show_default=True,
              help="Maximum order of the shim system, allowed values are 1, 2. Note that specifying 2 will return "
                   "orders 1 and 2")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(exists=True),
              default=__dir_config_scanner_constraints__, show_default=True,
              help="Constraints for the 1st and 2nd order scanner coils")
@click.option('--slices', type=click.Choice(['interleaved', 'sequential', 'volume']), required=False,
              default='sequential', show_default=True, help="Defines the slice ordering")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shim for 'interleaved' and 'sequential'")
@click.option('--optimizer-method', 'method', type=click.Choice(['least_squares', 'pseudo_inverse']), required=False,
              default='least_squares', show_default=True, help="Method used by the optimizer")
@click.option('--mask-dilation-kernel', 'dilation_kernel',
              type=click.Choice(['sphere', 'cross', 'line', 'cube', 'None']), required=False, default='sphere',
              show_default=True, help="Kernel used to dilate the mask to expand the roi")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False, default='3',
              show_default=True,
              help="Length of a side of the 3d kernel to dilate the mask. Must be odd. For example, a kernel of size 3"
                   "will dilate the mask by 1 pixel, 5->2 pixels")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True, help="Directory to output coil text file(s).")
@click.option('--output-format', 'o_format', type=click.Choice(['slicewise', 'shimwise']), default='slicewise',
              show_default=True, help="Format of the output txt file(s)")
def static_cli(fname_fmap, fname_anat, fname_mask_anat, method, slices, slice_factor, coils, dilation_kernel,
               dilation_kernel_size, scanner_coil_order, fname_sph_constr, path_output, o_format):
    """ Static shim by fitting a fieldmap. Example of use: st_shim fieldmap_static --coil coil1.nii
    coil1_constraints.json --coil coil2.nii coil2_constraints.json --fmap fmap.nii --anat anat.nii

    EXPAND

    """
    # Load the fieldmap
    nii_fmap_orig = nib.load(fname_fmap)

    # Make sure the fieldmap has the appropriate dimensions.
    if nii_fmap_orig.get_fdata().ndim != 3:
        raise ValueError("Fieldmap must be 3d (dim1, dim2, dim3)")

    # Extend the fieldmap if there are axes that are 1d. This is done since we are fitting a fieldmap to coil profiles,
    # having essentially a 2d matrix as a fieldmap can lead to errors in the through plane direction. To metigate this,
    # we create a 3d volume by replicating the single slice.
    if 1 in nii_fmap_orig.shape:
        n_slices_to_expand = int(math.ceil((dilation_kernel_size - 1) / 2))
        nii_fmap = _expand_fmap(nii_fmap_orig, n_slices_to_expand)

    # Load the anat
    nii_anat = nib.load(fname_anat)

    # Load mask
    if fname_mask_anat is not None:
        nii_mask_anat = nib.load(fname_mask_anat)
    else:
        # If no mask is provided, shim the whole anat volume
        nii_mask_anat = nib.Nifti1Image(np.ones_like(nii_anat.get_fdata()), nii_anat.affine, header=nii_anat.header)

    # Prepare the output
    create_output_dir(path_output)

    # Load the coils
    list_coils = []
    for coil in coils:
        nii_coil_profiles = nib.load(coil[0])
        constraints = json.load(coil[1])
        list_coils.append(Coil(nii_coil_profiles.get_fdata(), coil.affine, constraints))

    # Create the spherical harmonic coil profiles of the scanner
    if scanner_coil_order == 1 or scanner_coil_order == 2:

        mesh1, mesh2, mesh3 = generate_meshgrid(nii_fmap.shape, nii_fmap.affine)
        sph_coil_profile = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(range(1, scanner_coil_order + 1)))

        # It looks like for the prisma it would be 80mT/m --> 80000uT/m
        if os.path.isfile(fname_sph_constr):
            sph_contraints = json.load(open(fname_sph_constr))
        else:
            raise OSError("Missing json file")

        if scanner_coil_order == 1:
            # Order 1 only requires the first 3 channels
            sph_coil_profile = sph_coil_profile[..., :3]
            sph_contraints['coef_channel_minmax'] = sph_contraints['coef_channel_minmax'][:3]

        list_coils.append(Coil(sph_coil_profile, nii_fmap.affine, sph_contraints))

    if len(list_coils) == 0:
        raise RuntimeError("No custom or scanner coils were selected. Use --coil and/or --scanner-coil-order")

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    list_slices = define_slices(n_slices, slice_factor, slices)
    logger.info(f"The slices to shim are: {list_slices}")

    # Get shimming coefficients
    coefs = shim_sequencer(nii_fmap, nii_anat, nii_mask_anat, list_slices, list_coils, method=method,
                           mask_dilation_kernel=dilation_kernel, mask_dilation_kernel_size=dilation_kernel_size)

    # Output #
    end_channel = 0
    list_fname_output = []
    for i_coil in range(len(list_coils)):
        start_channel = end_channel
        coil = list_coils[i_coil]
        fname_output = os.path.join(path_output, f"coefs_coil{i_coil}_{coil.name}.txt")
        with open(fname_output, 'w', encoding='utf-8') as f:
            # (len(slices) x n_channels)
            n_channels = coil.dim[3]
            end_channel = start_channel + n_channels

            if o_format == 'shimwise':
                # Output per shim (chronological), output all channels for a particular shim, then repeat
                for i_shim in range(len(list_slices)):
                    for i_channel in range(n_channels):
                        f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}")
                        if i_channel != n_channels:
                            f.write(", ")
                    f.write("\n")

            elif o_format == 'slicewise':
                # Output per slice, output all channels for a particular slice, then repeat
                for i_slice in range(nii_anat.shape[2]):
                    for i_channel in range(n_channels):
                        i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                        f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}")
                        if i_channel != n_channels:
                            f.write(", ")
                    f.write("\n")
        list_fname_output.append(fname_output)

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")


def _expand_fmap(nii_fieldmap, n_exp_slices):
    """Expand the fieldmap for axis with dimension == 1"""
    fieldmap_shape = nii_fieldmap.shape
    list_axis = [i for i in range(len(fieldmap_shape)) if fieldmap_shape[i] == 1]
    for i_axis in list_axis:
        nii_fieldmap = extend_slice(nii_fieldmap, n_slices=n_exp_slices, axis=i_axis)

    return nii_fieldmap


@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option('--fmap', 'fname_fmap', required=True, type=click.Path(),
#               help="B0 fieldmap. This should be a 4d file (4th dimension being time")
# @click.option('--anat', 'fname_anat', type=click.Path(), required=True,
#               help="Filename of the anatomical image to apply the correction.")
# @click.option('--resp', 'fname_resp', type=click.Path(), required=True,
#               help="Siemens respiratory file containing pressure data.")
# @click.option('--mask-static', 'fname_mask_anat_static', type=click.Path(), required=False,
#               help="3D nifti file used to define the static spatial region to shim. "
#                    "The coordinate system should be the same as ``anat``'s coordinate system.")
# @click.option('--mask-riro', 'fname_mask_anat_riro', type=click.Path(), required=False,
#               help="3D nifti file used to define the time varying (i.e. RIRO, Respiration-Induced Resonance Offset) "
#                    "spatial region to shim. "
#                    "The coordinate system should be the same as ``anat``'s coordinate system.")
# @click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.curdir,
#               help="Directory to output gradient text file and figures.")
def realtime_cli():
    """ Realtime shim by fitting a fieldmap to a pressure monitoring unit."""
    a = 1


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--slices', required=True,
              help="Enter the total number of slices. Also accepts a path to an anatomical file to determine the "
                   "number of slices automatically. (Looks at 3rd dim)")
@click.option('--factor', required=True, type=click.INT,
              help="Number of slices per shim")
# Add 'volume'
@click.option('--method', type=click.Choice(['interleaved', 'sequential']), required=True,
              help="Defines how the slices should be sorted")
@click.option('-o', '--output', 'fname_output', type=click.Path(), default=os.path.join(os.curdir, 'slices.json'),
              show_default=True, help="Output filename for the json file")
def define_slices_cli(slices, factor, method, fname_output):
    """ Define slices to shim to a json file according to the number slices, factor and method used.

    """
    # Get the number of slices
    click.echo(type(slices))
    if os.path.isfile(slices):
        nii_anat = nib.load(slices)
        n_slices = nii_anat.shape[2]
    else:
        try:
            n_slices = int(slices)
        except ValueError:
            raise ValueError(f"Could not get the number of slices. Make sure {slices} is a number or a file that "
                             f"exists")

    list_slices = define_slices(n_slices, factor, method)

    if fname_output[-5:] != '.json':
        raise ValueError("Filename of the output must be a json file")
    create_output_dir(fname_output, is_file=True)

    with open(fname_output, 'w', encoding='utf-8') as f:
        json.dump(list_slices, f, ensure_ascii=False, indent=4)

    logger.info(f"The slices to shim are: {list_slices}")


shim_cli.add_command(realtime_shim_cli, 'gradient_realtime')
shim_cli.add_command(static_cli, 'fieldmap_static')
shim_cli.add_command(realtime_cli, 'fieldmap_realtime')
# shim_cli.add_command(define_slices_cli, 'define_slices')
