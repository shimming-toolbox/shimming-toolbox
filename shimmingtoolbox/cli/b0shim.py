# -*- coding: utf-8 -*-
"""
This file includes CLIs for shimming by fitting fieldmaps for static and realtime shimming. It groups them along with
the gradient method in a st_shim CLI with the argument being:
- fieldmap_static
- fieldmap_realtime
- gradient_realtime
"""

import click
import os
import nibabel as nib
import numpy as np
import logging
import json
import math

from shimmingtoolbox import __dir_config_scanner_constraints__
from shimmingtoolbox.cli.realtime_shim import realtime_shim_cli
from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.coils.coordinates import generate_meshgrid, phys_to_vox_coefs
from shimmingtoolbox.coils.siemens_basis import siemens_basis
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.shim.sequencer import shim_sequencer, shim_realtime_pmu_sequencer, new_bounds_from_currents
from shimmingtoolbox.shim.sequencer import extend_slice, define_slices
from shimmingtoolbox.utils import create_output_dir, set_all_loggers


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Shim according to the specified algorithm as an argument e.g. st_shim xxxxx")
def b0shim_cli():
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
              help="Nifti filename of the B0 fieldmap. This file should contain a 3d array.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Nifti filename of the anatomical image to apply the correction onto."
                   "This file should contain a 3d array.")
@click.option('--mask', 'fname_mask_anat', type=click.Path(exists=True), required=False,
              help="Nifti file used to define the spatial region to shim. This file should contain a 3d array."
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('--scanner-coil-order', type=click.INT, default=-1, show_default=True,
              help="Maximum order of the shim system, allowed values are 0, 1, 2. Note that specifying 2 will return "
                   "orders 0, 1 and 2. The 0th order is the f0 frequency.")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(exists=True),
              default=__dir_config_scanner_constraints__, show_default=True,
              help="Constraints for the 1st and 2nd order scanner coils")
@click.option('--slices', type=click.Choice(['interleaved', 'sequential', 'volume']), required=False,
              default='sequential', show_default=True, help="Defines the slice ordering")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shim for 'interleaved' and 'sequential'")
@click.option('--optimizer-method', 'method', type=click.Choice(['least_squares', 'pseudo_inverse']), required=False,
              default='least_squares', show_default=True, help="Method used by the optimizer. LS will respect the "
                                                               "constraints, PS will not respect the constraints")
@click.option('--mask-dilation-kernel', 'dilation_kernel',
              type=click.Choice(['sphere', 'cross', 'line', 'cube', 'None']), required=False, default='sphere',
              show_default=True, help="Kernel used to dilate the mask to expand the roi")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False, default='3',
              show_default=True,
              help="Length of a side of the 3d kernel to dilate the mask. Must be odd. For example, a kernel of size 3"
                   "will dilate the mask by 1 pixel, 5->2 pixels")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True, help="Directory to output coil text file(s).")
@click.option('--output-file-format-coil', 'o_format_coil',
              type=click.Choice(['slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil']),
              default='slicewise-coil',
              show_default=True, help="Format of the output txt file(s) for the custom coils. slicewise will output "
                                      "one slice per row in the txt file, chronological will output one set of shim "
                                      "per row in the order that the shim will be performed. Use 'ch' or 'coil' to "
                                      "specify whether to output one txt file per coil system or coil channel.")
@click.option('--output-file-format-scanner', 'o_format_sph',
              type=click.Choice(['slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil']),
              default='slicewise-coil',
              show_default=True, help="Format of the output txt file(s) for the custom coils. slicewise will output "
                                      "one slice per row in the txt file, chronological will output one set of shim "
                                      "per row in the order that the shim will be performed. Use 'ch' or 'coil' to "
                                      "specify whether to output one txt file per coil system or coil channel.")
@click.option('--output-value-format', 'output_value_format', type=click.Choice(['delta', 'absolute']), default='delta',
              help="Format of the scanner coil output. Delta: Outputs the change of coefficients. Absolute: Outputs "
                   "the coefficient directly by taking into account the current shim settings. This is effectively "
                   "initial + delta")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def static_cli(fname_fmap, fname_anat, fname_mask_anat, method, slices, slice_factor, coils, dilation_kernel,
               dilation_kernel_size, scanner_coil_order, fname_sph_constr, path_output, o_format_coil, o_format_sph,
               output_value_format, verbose):
    """ Static shim by fitting a fieldmap. Use the option --optimizer-method to change the shimming algorithm used to
    optimize. Use the options --slices and --slice-factor to change the shimming order/size of the slices.

    Example of use: st_shim fieldmap_static --coil coil1.nii coil1_config.json
    --coil coil2.nii coil2_config.json --fmap fmap.nii --anat anat.nii --mask mask.nii
    """
    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(path_output)

    # Load the fieldmap, expand the dimensions of the fieldmap if one of the dimensions is 2 or less. This is done since
    # we are fitting a fieldmap to coil profiles, having essentially a 2d matrix as a fieldmap can lead to errors in the
    # through plane direction.
    fmap_required_dims = 3
    nii_fmap = _load_fmap(fname_fmap, fmap_required_dims, dilation_kernel_size, path_output)

    # Load the anat
    nii_anat = nib.load(fname_anat)
    dim_info = nii_anat.header.get_dim_info()
    if dim_info[2] != 2:
        # Slice must be the 3rd dimension of the file
        # TODO: Reorient nifti so that the slice is the 3rd dim
        raise RuntimeError("Slice encode direction must be the 3rd dimension of the nifti")

    # Load mask
    if fname_mask_anat is not None:
        nii_mask_anat = nib.load(fname_mask_anat)
    else:
        # If no mask is provided, shim the whole anat volume
        nii_mask_anat = nib.Nifti1Image(np.ones_like(nii_anat.get_fdata()), nii_anat.affine, header=nii_anat.header)

    if logger.level <= getattr(logging, 'DEBUG'):
        # Save inputs
        list_fname = [
            fname_fmap,
            fname_anat,
            fname_mask_anat
        ]
        _save_nii_to_new_dir(list_fname, path_output)

    # Get current coefs
    fname_json = fname_fmap.split('.nii')[0] + '.json'
    initial_coefs = _get_current_shim_settings(fname_json)

    # Load the coils
    list_coils = _load_coils(coils, scanner_coil_order, fname_sph_constr, nii_fmap, initial_coefs)

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    list_slices = define_slices(n_slices, slice_factor, slices)
    logger.info(f"The slices to shim are:\n{list_slices}")

    # Get shimming coefficients
    coefs = shim_sequencer(nii_fmap, nii_anat, nii_mask_anat, list_slices, list_coils,
                           method=method,
                           mask_dilation_kernel=dilation_kernel,
                           mask_dilation_kernel_size=dilation_kernel_size,
                           path_output=path_output)

    if output_value_format == 'absolute':
        raise NotImplementedError("absolute not yet implemented")
        # TODO: Returned values change depending on the scanner as well as the shim order.
        #  Ryan has the units sorted out for the prisma fit
        # https://github.com/shimming-toolbox/shimming-toolbox-matlab/blob/master/Coils/Shim_Siemens/Shim_Prisma/Shim_IUGM_Prisma_fit/ShimSpecs_IUGM_Prisma_fit.m

        # order_mapping = {0: 1,
        #                  1: 4,
        #                  2: 9}
        # n_channels = order_mapping[scanner_coil_order]
        # for i_channel in range(n_channels):
        #     coefs[:, -n_channels + i_channel] = coefs[:, -n_channels + i_channel] + initial_coefs[i_channel]

    # Output
    if scanner_coil_order >= 0:
        n_channels = list_coils[-1].dim[3]

        if scanner_coil_order >= 1:
            # TODO: Fix for 2nd order
            # Convert coef of 1st order sph harmonics to voxel coord system

            # offset by 5 channels if using 2nd order
            if scanner_coil_order == 2:
                offset = 5
            else:
                offset = 0

            # Invert coefficients of the 1st order, scanner shim coefficients (LAI) --> NIfTI coefficients (RAS)
            # x
            coefs[..., -3 - offset] = -coefs[..., -3 - offset]
            # z
            coefs[..., -1 - offset] = -coefs[..., -1 - offset]

            # Convert from patient coordinates to image coordinates
            scanner_coil_coef_vox = phys_to_vox_coefs(coefs[..., -3 - offset], coefs[..., -2 - offset],
                                                      coefs[..., -1 - offset], nii_anat.affine)
            coefs[..., -3 - offset] = scanner_coil_coef_vox[0]
            coefs[..., -2 - offset] = scanner_coil_coef_vox[1]
            coefs[..., -1 - offset] = scanner_coil_coef_vox[2]

            # Convert from image to freq, phase, slice encoding direction
            logger.debug("Converting scanner coil from voxel x, y, z to freq, phase and slice encoding direction")
            dim_info = nii_anat.header.get_dim_info()
            order1 = coefs[..., -3 - offset:coefs.shape[-1] - offset]
            curr_freq, curr_phase, curr_slice = [order1[..., dim] for dim in dim_info]

            # TODO: Phase encode direction
            coefs[..., -3 - offset] = curr_freq
            coefs[..., -2 - offset] = curr_phase
            coefs[..., -1 - offset] = curr_slice

        list_fname_output = _save_to_text_file_static(list_coils[-1:], coefs[..., -n_channels:], list_slices,
                                                      path_output, o_format_sph)
        if len(list_coils) > 1:
            fname_tmp = _save_to_text_file_static(list_coils[:-1], coefs[..., :-n_channels], list_slices, path_output,
                                                  o_format_coil, start_coil_number=1)
            # Concat list
            list_fname_output = list_fname_output + fname_tmp
    else:
        # The case where there is no custom coil or scanner coil is already checked in load_coils so no need to check
        # again i.e. there must be a custom coil at this point
        list_fname_output = _save_to_text_file_static(list_coils, coefs, list_slices, path_output, o_format_coil)

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")


def _save_to_text_file_static(list_coils, coefs, list_slices, path_output, o_format, start_coil_number=0):
    """o_format can either be 'slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil'"""

    end_channel = 0
    list_fname_output = []
    for i_coil in range(len(list_coils)):
        start_channel = end_channel
        coil = list_coils[i_coil]
        n_channels = coil.dim[3]
        end_channel = start_channel + n_channels

        if o_format[-5:] == '-coil':

            fname_output = os.path.join(path_output, f"coefs_coil{start_coil_number + i_coil}_{coil.name}.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                # (len(slices) x n_channels)

                if o_format == 'chronological-coil':
                    # Output per shim (chronological), output all channels for a particular shim, then repeat
                    for i_shim in range(len(list_slices)):
                        for i_channel in range(n_channels):
                            f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}")
                            if i_channel != n_channels:
                                f.write(", ")
                        f.write("\n")

                elif o_format == 'slicewise-coil':
                    # Output per slice, output all channels for a particular slice, then repeat
                    # Assumes all slices are in list_slices once which is the case for sequential, interleaved and
                    # volume
                    n_slices = np.sum([len(a_shim) for a_shim in list_slices])
                    for i_slice in range(n_slices):
                        i_shim = [list_slices.index(a_shim) for a_shim in list_slices if i_slice in a_shim][0]
                        for i_channel in range(n_channels):
                            f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}")
                            if i_channel != n_channels:
                                f.write(", ")
                        f.write("\n")

                list_fname_output.append(os.path.abspath(fname_output))

        else:
            # o_format[-3:] == '-ch':
            # Write a file for each channel
            for i_channel in range(n_channels):
                fname_output = os.path.join(path_output, f"coefs_coil{i_coil}_ch{i_channel}_{coil.name}.txt")

                if o_format == 'chronological-ch':
                    with open(fname_output, 'w', encoding='utf-8') as f:
                        # Each row will have one coef representing the shim in chronological order
                        for i_shim in range(len(list_slices)):
                            f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}\n")

                if o_format == 'slicewise-ch':
                    with open(fname_output, 'w', encoding='utf-8') as f:
                        # Each row will have one coef representing the shim in slicewise order
                        n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                        for i_slice in range(n_slices):
                            i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                            f.write(f"{coefs[i_shim, start_channel + i_channel]:.6f}\n")

                list_fname_output.append(os.path.abspath(fname_output))

    return list_fname_output


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils', nargs=2, multiple=True, type=(click.Path(exists=True), click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__dir_config_scanner_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Nifti filename of the B0 fieldmap. This file should contain a 4d array. 4th diension being the "
                   "time.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Nifti filename of the anatomical image to apply the correction onto."
                   "This file should contain a 3d array.")
@click.option('--resp', 'fname_resp', type=click.Path(exists=True), required=True,
              help="Siemens respiratory file containing pressure data.")
@click.option('--mask-static', 'fname_mask_anat_static', type=click.Path(exists=True), required=False,
              help="Nifti file used to define the static region to shim. This file should contain a 3d array."
                   "The coordinate system should be the same as ``anat``'s coordinate system.")
@click.option('--mask-riro', 'fname_mask_anat_riro', type=click.Path(exists=True), required=False,
              help="Nifti file used to define the time varying (i.e. RIRO, Respiration-Induced Resonance Offset) "
                   "region to shim. This file should contain a 3d array. The coordinate system should be the same as "
                   "``anat``'s coordinate system.")
@click.option('--scanner-coil-order', type=click.INT, default=-1, show_default=True,
              help="Maximum order of the shim system, allowed values are 0, 1, 2. Note that specifying 2 will return "
                   "orders 0, 1 and 2. The 0th order is the f0 frequency.")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(exists=True),
              default=__dir_config_scanner_constraints__, show_default=True,
              help="Constraints for the 1st and 2nd order scanner coils")
@click.option('--slices', type=click.Choice(['interleaved', 'sequential', 'volume']), required=False,
              default='sequential', show_default=True, help="Defines the slice ordering")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shim for 'interleaved' and 'sequential'")
@click.option('--optimizer-method', 'method', type=click.Choice(['least_squares', 'pseudo_inverse']), required=False,
              default='least_squares', show_default=True, help="Method used by the optimizer. LS will respect the "
                                                               "constraints, PS will not respect the constraints")
@click.option('--mask-dilation-kernel', 'dilation_kernel',
              type=click.Choice(['sphere', 'cross', 'line', 'cube', 'None']), required=False, default='sphere',
              show_default=True, help="Kernel used to dilate the mask to expand the roi")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False, default='3',
              show_default=True,
              help="Length of a side of the 3d kernel to dilate the mask. Must be odd. For example, a kernel of size 3"
                   "will dilate the mask by 1 pixel, 5->2 pixels")
@click.option('--output-file-format', 'o_format', type=click.Choice(['slicewise-ch', 'chronological-ch', 'eva']),
              default='slicewise-ch',
              show_default=True, help="Format of the output txt file(s) of the coils. slicewise will output "
                                      "one slice per row in the txt file, chronological will output one set of shim "
                                      "per row in the order that the shim will be performed.")
@click.option('--output-value-format', 'output_value_format', type=click.Choice(['delta', 'absolute']),
              default='delta', show_default=True,
              help="Format of the scanner coil output. Delta: Outputs the change of coefficients. Absolute: Outputs "
                   "the coefficient directly by taking into account the current shim settings. This is effectively "
                   "initial + delta")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True, help="Directory to output coil text file(s).")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def realtime_cli(fname_fmap, fname_anat, fname_mask_anat_static, fname_mask_anat_riro, fname_resp, method, slices,
                 slice_factor, coils, dilation_kernel, dilation_kernel_size, scanner_coil_order, fname_sph_constr,
                 path_output, o_format, output_value_format, verbose):
    """ Realtime shim by fitting a fieldmap to a pressure monitoring unit. Use the option --optimizer-method to change
    the shimming algorithm used to optimize. Use the options --slices and --slice-factor to change the shimming
    order/size of the slices.

    Example of use: st_shim fieldmap_static --coil coil1.nii coil1_config.json
    --coil coil2.nii coil2_config.json --fmap fmap.nii --anat anat.nii --mask-static mask.nii
    """
    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(path_output)

    # Load the fieldmap, expand the dimensions of the fieldmap if one of the dimensions is 2 or less. This is done since
    # we are fitting a fieldmap to coil profiles, having essentially a 2d matrix as a fieldmap can lead to errors in the
    # through plane direction.
    fmap_required_dims = 4
    nii_fmap = _load_fmap(fname_fmap, fmap_required_dims, dilation_kernel_size, path_output)

    # Load json associated with the fieldmap
    fname_json = fname_fmap.rsplit('.nii', 1)[0] + '.json'
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Load the anat
    nii_anat = nib.load(fname_anat)
    dim_info = nii_anat.header.get_dim_info()
    if dim_info[2] != 2:
        # Slice must be the 3rd dimension of the file
        # TODO: Reorient nifti so that the slice is the 3rd dim
        raise RuntimeError("Slice encode direction must be the 3rd dimension of the nifti")

    # Load static mask
    if fname_mask_anat_static is not None:
        nii_mask_anat_static = nib.load(fname_mask_anat_static)
    else:
        # If no mask is provided, shim the whole anat volume
        nii_mask_anat_static = nib.Nifti1Image(np.ones_like(nii_anat.get_fdata()), nii_anat.affine,
                                               header=nii_anat.header)

    # Load riro mask
    if fname_mask_anat_riro is not None:
        nii_mask_anat_riro = nib.load(fname_mask_anat_riro)
    else:
        # If no mask is provided, shim the whole anat volume
        nii_mask_anat_riro = nib.Nifti1Image(np.ones_like(nii_anat.get_fdata()), nii_anat.affine,
                                             header=nii_anat.header)

    # Get current coefs
    fname_json = fname_fmap.split('.nii')[0] + '.json'
    initial_coefs = _get_current_shim_settings(fname_json)

    # Load the coils
    list_coils = _load_coils(coils, scanner_coil_order, fname_sph_constr, nii_fmap, initial_coefs)

    if logger.level <= getattr(logging, 'DEBUG'):
        # Save inputs
        list_fname = [
            fname_fmap,
            fname_anat,
            fname_mask_anat_static,
            fname_mask_anat_riro
        ]
        _save_nii_to_new_dir(list_fname, path_output)

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    list_slices = define_slices(n_slices, slice_factor, slices)
    logger.info(f"The slices to shim are: {list_slices}")

    # Load PMU
    pmu = PmuResp(fname_resp)

    out = shim_realtime_pmu_sequencer(nii_fmap, json_data, nii_anat, nii_mask_anat_static, nii_mask_anat_riro,
                                      list_slices, pmu, list_coils,
                                      opt_method=method,
                                      mask_dilation_kernel=dilation_kernel,
                                      mask_dilation_kernel_size=dilation_kernel_size,
                                      path_output=path_output)

    currents_static, currents_riro, mean_p, p_rms = out

    if output_value_format == 'absolute':
        raise NotImplementedError("absolute not yet implemented")
        # TODO: Returned values change depending on the scanner as well as the shim order.
        #  Ryan has the units sorted out for the prisma fit
        # https://github.com/shimming-toolbox/shimming-toolbox-matlab/blob/master/Coils/Shim_Siemens/Shim_Prisma/Shim_IUGM_Prisma_fit/ShimSpecs_IUGM_Prisma_fit.m

        # order_mapping = {0: 1,
        #                  1: 4,
        #                  2: 9}
        # n_channels = order_mapping[scanner_coil_order]
        # for i_channel in range(n_channels):
        #     currents_static[:, -n_channels + i_channel] = currents_static[:, -n_channels + i_channel] + \
        #                                                   initial_coefs[i_channel]

    # Output
    # TODO: Fix for 2nd order

    if scanner_coil_order >= 0:
        if scanner_coil_order >= 1:
            # offset by 5 channels if using 2nd order
            if scanner_coil_order == 2:
                offset = 5
            else:
                offset = 0

            logger.debug("Converting scanner coil from phys x, y, z to voxel x, y, z")
            # Invert coefficients of the 1st order, scanner shim coefficients (LAI) --> NIfTI coefficients (RAS)
            # x
            currents_static[..., -3 - offset] = -currents_static[..., -3 - offset]
            currents_riro[..., -3 - offset] = -currents_riro[..., -3 - offset]
            # z
            currents_static[..., -1 - offset] = -currents_static[..., -1 - offset]
            currents_riro[..., -1 - offset] = -currents_riro[..., -1 - offset]

            # Convert static from patient to image coord system
            scanner_coil_coef_vox = phys_to_vox_coefs(currents_static[..., -3 - offset],
                                                      currents_static[..., -2 - offset],
                                                      currents_static[..., -1 - offset], nii_anat.affine)
            currents_static[..., -3 - offset] = scanner_coil_coef_vox[0]
            currents_static[..., -2 - offset] = scanner_coil_coef_vox[1]
            currents_static[..., -1 - offset] = scanner_coil_coef_vox[2]

            # Convert riro to voxel coord system
            scanner_coil_coef_vox = phys_to_vox_coefs(currents_riro[..., -3 - offset], currents_riro[..., -2 - offset],
                                                      currents_riro[..., -1 - offset], nii_anat.affine)
            currents_riro[..., -3 - offset] = scanner_coil_coef_vox[0]
            currents_riro[..., -2 - offset] = scanner_coil_coef_vox[1]
            currents_riro[..., -1 - offset] = scanner_coil_coef_vox[2]

            # Convert from image to freq, phase, slice encoding direction
            logger.debug("Converting scanner coil from voxel x, y, z to freq, phase and slice encoding direction")
            # TODO: Phase encode direction
            dim_info = nii_anat.header.get_dim_info()
            # static
            order1_static = currents_static[..., -3 - offset:currents_static.shape[-1] - offset]
            curr_static_freq, curr_static_phase, curr_static_slice = [order1_static[..., dim] for dim in dim_info]
            currents_static[..., -3 - offset] = curr_static_freq
            currents_static[..., -2 - offset] = curr_static_phase
            currents_static[..., -1 - offset] = curr_static_slice
            # riro
            order1_riro = currents_riro[..., -3 - offset:currents_riro.shape[-1] - offset]
            curr_riro_freq, curr_riro_phase, curr_riro_slice = [order1_riro[..., dim] for dim in dim_info]
            currents_riro[..., -3 - offset] = curr_riro_freq
            currents_riro[..., -2 - offset] = curr_riro_phase
            currents_riro[..., -1 - offset] = curr_riro_slice

    _save_to_text_file_rt(list_coils, currents_static, currents_riro, mean_p, list_slices, path_output, o_format)


def _save_to_text_file_rt(list_coils, currents_static, currents_riro, mean_p, list_slices, path_output, o_format):
    """o_format can either be 'chronological-ch', 'chronological-coil'"""

    end_channel = 0
    list_fname_output = []
    n_coils = len(list_coils)
    for i_coil in range(n_coils):
        start_channel = end_channel
        coil = list_coils[i_coil]
        n_channels = coil.dim[3]
        end_channel = start_channel + n_channels

        # o_format[-3:] == '-ch':
        # Write a file for each channel
        for i_channel in range(n_channels):
            fname_output = os.path.join(path_output, f"coefs_coil{i_coil}_ch{i_channel}_{coil.name}.txt")

            if o_format == 'chronological-ch':
                with open(fname_output, 'w', encoding='utf-8') as f:
                    # Each row will have 3 coef representing the static, riro and mean_p in chronological order
                    for i_shim in range(len(list_slices)):
                        f.write(f"{currents_static[i_shim, start_channel + i_channel]:.6f}, ")
                        f.write(f"{currents_riro[i_shim, start_channel + i_channel]:.12f}, ")
                        f.write(f"{mean_p:.4f}\n")

            if o_format == 'slicewise-ch':
                with open(fname_output, 'w', encoding='utf-8') as f:
                    # Each row will have one coef representing the static, riro and mean_p in slicewise order
                    n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                    for i_slice in range(n_slices):
                        i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                        f.write(f"{currents_static[i_shim, start_channel + i_channel]:.6f}, ")
                        f.write(f"{currents_riro[i_shim, start_channel + i_channel]:.12f}, ")
                        f.write(f"{mean_p:.4f}\n")

            # TODO: Remove once implemented in more streamlined way
            if o_format == 'eva':

                # Make sure there are 4 channels
                if n_channels != 4:
                    raise RuntimeError("Eva's output format should only be used with 1st order scanner coils")

                name = {0: 'f0',
                        1: 'x',
                        2: 'y',
                        3: 'z'}

                fname_output = os.path.join(path_output, f"{name[i_channel]}shim_gradients.txt")
                with open(fname_output, 'w', encoding='utf-8') as f:
                    n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                    for i_slice in range(n_slices):
                        i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                        # Divide by 1000 for mt/m units
                        f.write(f"corr_vec[0][{i_slice}]= "
                                f"{currents_static[i_shim, start_channel + i_channel] / 1000:.6f}\n")
                        f.write(f"corr_vec[1][{i_slice}]= "
                                f"{currents_riro[i_shim, start_channel + i_channel] / 1000:.12f}\n")
                        f.write(f"corr_vec[2][{i_slice}]= {mean_p:.3f}\n")

            list_fname_output.append(os.path.abspath(fname_output))

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")


def _load_fmap(fname_fmap, n_dims, dilation_kernel_size, path_output):
    """ Load the fmap and expand its dimensions to the kernel size

    Args:
        fname_fmap (str): Filename of the fieldmap
        n_dims (int): Number of dimensions of the fieldmap (3 or 4)
        dilation_kernel_size: Size of the kernel

    Returns:
        nibabel.Nifti1Image: Nibabel object of the loaded and extended fieldmap

    """
    # Load the fieldmap
    nii_fmap_orig = nib.load(fname_fmap)

    # Make sure the fieldmap has the appropriate dimensions.
    if nii_fmap_orig.get_fdata().ndim != n_dims:
        raise ValueError(f"Fieldmap must be {n_dims}")

    # Extend the fieldmap if there are axes that are 1d. This is done since we are fitting a fieldmap to coil profiles,
    # having essentially a 2d matrix as a fieldmap can lead to errors in the through plane direction. To metigate this,
    # we create a 3d volume by replicating the single slice.
    if 1 in nii_fmap_orig.shape[:3]:
        n_slices_to_expand = int(math.ceil((dilation_kernel_size - 1) / 2))
        fieldmap_shape = nii_fmap_orig.shape
        # Find the list of axes that has a length of 1
        list_axis = [i for i in range(3) if fieldmap_shape[i] == 1]

        # Extend for each axes
        tmp_nii = nii_fmap_orig
        for i_axis in list_axis:
            tmp_nii = extend_slice(tmp_nii, n_slices=n_slices_to_expand, axis=i_axis)
        nii_fmap = tmp_nii

        # If DEBUG, save the extended fieldmap
        if logger.level <= getattr(logging, 'DEBUG'):
            fname_new_fmap = os.path.join(path_output, 'tmp_extended_fmap.nii.gz')
            nib.save(nii_fmap, fname_new_fmap)
            logger.debug(f"Extended fmap, saved the new fieldmap here: {fname_new_fmap}")

    else:
        # Load the original
        nii_fmap = nii_fmap_orig

    return nii_fmap


def _load_coils(coils, order, fname_constraints, nii_fmap, initial_coefs):
    """ Loads the Coil objects from filenames

    Args:
        coils (list): List of tuples(fname_nii, fname_json) os coil profiles and constraints
        order (int): Order of the scanner coils (0 or 1 or 2)
        fname_constraints (str): Filename of the constraints of the scanner coils
        nii_fmap (nib.Nifti1Image): Nibabel object of the fieldmap
        initial_coefs (list): 1d array of the initial coefficients of the scanner coil profiles

    Returns:
        list: List of Coil objects containing the custom coils followed by the scanner coil if requested
    """
    list_coils = []

    # Load custom coils
    for coil in coils:
        nii_coil_profiles = nib.load(coil[0])
        constraints = json.load(open(coil[1]))
        list_coils.append(Coil(nii_coil_profiles.get_fdata(), nii_coil_profiles.affine, constraints))

    # Create the spherical harmonic coil profiles of the scanner
    if 0 <= order <= 2:

        # Define profile for Tx (constant volume)
        profile_order_0 = np.ones(nii_fmap.shape[:3])

        # define the coil profiles
        if order == 0:
            # f0 --> [1]
            sph_coil_profile = profile_order_0[..., np.newaxis]
        else:
            # f0, orders
            mesh1, mesh2, mesh3 = generate_meshgrid(nii_fmap.shape[:3], nii_fmap.affine)
            profile_orders = siemens_basis(mesh1, mesh2, mesh3, orders=tuple(range(1, order + 1)))
            sph_coil_profile = np.concatenate((profile_order_0[..., np.newaxis], profile_orders), axis=3)

        if os.path.isfile(fname_constraints):
            sph_contraints = json.load(open(fname_constraints))

            def _initial_in_bounds(coefs, bounds):
                """Makes sure the initial values are within the bounds of the constraints"""
                if len(coefs) != len(bounds):
                    raise RuntimeError("The scanner coil's bounds is not the same length as the initial bounds found "
                                       "in the json")
                for i_bound in range(len(bounds)):
                    if not (bounds[i_bound][0] <= coefs[i_bound] <= bounds[i_bound][1]):
                        raise RuntimeError(f"Initial scanner coefs are outside the bounds allowed in the constraints: "
                                           f"{bounds[i_bound]}, initial: {coefs[i_bound]}")

            # TODO: Implement once units are sorted out
            # _initial_in_bounds(initial_coefs, sph_contraints['coef_channel_minmax'])
            sph_contraints['coef_channel_minmax'] = new_bounds_from_currents(np.array([initial_coefs]),
                                                                             sph_contraints['coef_channel_minmax'])[0]
        else:
            raise OSError("Missing json file")

        # Restrict constraint coefficient size/bounds depending on the order
        if order == 0:
            # f0 --> [1]
            sph_coil_profile = sph_coil_profile[..., :1]
            sph_contraints['coef_channel_minmax'] = sph_contraints['coef_channel_minmax'][:1]
        # f0, x, y, z -- > [4]
        elif order == 1:
            # Order 1 only requires the first 3 channels + Tx
            sph_contraints['coef_channel_minmax'] = sph_contraints['coef_channel_minmax'][:4]

        list_coils.append(Coil(sph_coil_profile, nii_fmap.affine, sph_contraints))

    # Make sure a coil is selected
    if len(list_coils) == 0:
        raise RuntimeError("No custom or scanner coils were selected. Use --coil and/or --scanner-coil-order")

    return list_coils


def _save_nii_to_new_dir(list_fname, path_output):
    """List of nii to save to a new output folder"""
    logger.debug(f"Saving CLI inputs to: {path_output}")
    for fname in list_fname:
        if fname is None:
            continue
        nii = nib.load(fname)
        fname_to_save = os.path.join(path_output, os.path.basename(fname))
        nib.save(nii, fname_to_save)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--slices', required=True,
              help="Enter the total number of slices. Also accepts a path to an anatomical file to determine the "
                   "number of slices automatically. (Looks at 3rd dim)")
@click.option('--factor', required=True, type=click.INT,
              help="Number of slices per shim")
@click.option('--method', type=click.Choice(['interleaved', 'sequential', 'volume']), required=True,
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


def _get_current_shim_settings(fname_json):

    # Read from json file
    if os.path.isfile(fname_json):
        json_data = json.load(open(fname_json))
    else:
        raise OSError("Missing json file")

    # Get the current coefficients of the spherical harmonics coil profiles
    current_coefs = json_data['ShimSetting']
    f0 = json_data['ImagingFrequency'] * 1e6
    # Tx (1) + 1st order (3) + 2nd order (5)
    current_coefs.insert(0, int(f0))

    return current_coefs


b0shim_cli.add_command(realtime_shim_cli, 'gradient_realtime')
b0shim_cli.add_command(static_cli, 'fieldmap_static')
b0shim_cli.add_command(realtime_cli, 'fieldmap_realtime')
# shim_cli.add_command(define_slices_cli, 'define_slices')
