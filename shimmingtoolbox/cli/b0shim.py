# -*- coding: utf-8 -*-

"""
This file includes CLIs for shimming by fitting fieldmaps for static and realtime shimming. It groups them along with
the gradient method in a st_shim CLI with the argument being:
- fieldmap_static
- fieldmap_realtime
- gradient_realtime
"""

import click
import copy
import json
import nibabel as nib
import numpy as np
import logging
import os
from matplotlib.figure import Figure

from shimmingtoolbox import __dir_config_scanner_constraints__
from shimmingtoolbox.cli.realtime_shim import gradient_realtime
from shimmingtoolbox.coils.coil import Coil, ScannerCoil, get_scanner_constraints, restrict_sph_constraints
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.shim.sequencer import ShimSequencer, RealTimeSequencer
from shimmingtoolbox.shim.sequencer import shim_max_intensity, define_slices
from shimmingtoolbox.shim.sequencer import extend_fmap_to_kernel_size, parse_slices, new_bounds_from_currents
from shimmingtoolbox.utils import create_output_dir, set_all_loggers, timeit
from shimmingtoolbox.shim.shim_utils import phys_to_gradient_cs, shim_to_phys_cs
from shimmingtoolbox.shim.shim_utils import ScannerShimSettings

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AVAILABLE_ORDERS = [-1, 0, 1, 2]


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Shim according to the specified algorithm as an argument e.g. st_b0shim xxxxx")
def b0shim_cli():
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils', nargs=2, multiple=True, type=(click.Path(exists=True), click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If using the scanner's gradient/shim coils, the coil profiles "
                   "must be in Hz/unit_shim and fieldmaps must be in Hz. If you want to shim using the scanner's "
                   "gradient/shim coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__dir_config_scanner_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Static B0 fieldmap.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Anatomical image to apply the correction onto.")
@click.option('--mask', 'fname_mask_anat', type=click.Path(exists=True), required=False,
              help="Mask defining the spatial region to shim.")
@click.option('--scanner-coil-order', 'scanner_coil_order', type=click.STRING, default='-1', show_default=True,
              help="Spherical harmonics orders to be used in optimization. "
                   f"Available orders: {AVAILABLE_ORDERS}. "
                   "Orders should be writen with a coma separating the values. (i.e. 0,1,2)"
                   "The 0th order is the f0 frequency.")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(), default="",
              help=f"Constraints for the scanner coil. Example file located: {__dir_config_scanner_constraints__}")
@click.option('--slices', type=click.Choice(['interleaved', 'sequential', 'volume', 'auto']), required=False,
              default='auto', show_default=True,
              help="Define the slice ordering. If set to 'auto', automatically parse the target image.")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shimmed group. Used when '--slices' is not set to 'auto'. For example, if the "
                   "'--slice-factor' value is '3', then with the 'sequential' mode, shimming will be performed "
                   "independently on the following groups: {0,1,2}, {3,4,5}, etc. With the mode 'interleaved', "
                   "it will be: {0,2,4}, {1,3,5}, etc.")
@click.option('--optimizer-method', 'method', required=False, default='quad_prog', show_default=True,
              type=click.Choice(['least_squares', 'pseudo_inverse', 'quad_prog', 'gradient']),
              help="Method used by the optimizer. LS, QP will respect the constraints, "
              "gradient method only accepts bounds for each channel "
              "PS will not respect the constraints")
@click.option('--regularization-factor', 'reg_factor', type=click.FLOAT, required=False, default=0.0, show_default=True,
              help="Regularization factor for the current when optimizing. A higher coefficient will penalize higher "
                   "current values while 0 provides no regularization. Not relevant for 'pseudo-inverse' "
                   "optimizer_method.")
@click.option('--optimizer-criteria', 'opt_criteria', type=click.Choice(['mse', 'mae', 'ps_huber']), required=False,
              default='mse', show_default=True,
              help="Criteria of optimization for the optimizer 'least_squares' and 'gradient'."
                   " mse: Mean Squared Error, mae: Mean Absolute Error, ps-huber: pseudo huber cost function ")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False, default='3',
              show_default=True,
              help="Number of voxels to consider outside of the masked area. For example, when doing dynamic shimming "
                   "with a linear gradient, the coefficient corresponding to the gradient orthogonal to a single "
                   "slice cannot be estimated: there must be at least 2 (ideally 3) points to properly estimate the "
                   "linear term. When using 2nd order or more, more dilation is necessary.")
@click.option('--fatsat', type=click.Choice(['auto', 'yes', 'no']), default='auto', show_default=True,
              help="Describe what to do with a fat saturation pulse. 'auto': It will parse the NIfTI file "
                   "for a fat-sat pulse and add shim coefficients of 0s before every shim group when using "
                   "'chronological-...' output-file-format-coil. 'no': It will not add 0s. 'yes': It will add 0s.")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True, help="Directory to output coil text file(s).")
@click.option('--output-file-format-coil', 'o_format_coil',
              type=click.Choice(['slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil']),
              default='slicewise-coil',
              show_default=True, help="Syntax used to describe the sequence of shim events for custom coils. "
                                      "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                                      "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                                      "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                                      "captured by the controller of the shim amplifier. Use 'ch' to output one "
                                      "file per coil channel (coil1_ch1.txt, coil1_ch2.txt, etc.). Use 'coil' to "
                                      "output one file per coil system (coil1.txt, coil2.txt). In the latter case, "
                                      "all coil channels are encoded across multiple columns in the text file.")
@click.option('--output-file-format-scanner', 'o_format_sph',
              type=click.Choice(['slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil',
                                 'gradient']),
              default='slicewise-coil',
              show_default=True, help="Syntax used to describe the sequence of shim events for scanner coils. "
                                      "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                                      "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                                      "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                                      "captured by the controller of the shim amplifier. If there is a fat saturation "
                                      "pulse in the anat sequence, shim weights of 0s are included in the output "
                                      "text file before each slice coefficients. Use 'ch' to output one "
                                      "file per coil channel (coil1_ch1.txt, coil1_ch2.txt, etc.). Use 'coil' to "
                                      "output one file per coil system (coil1.txt, coil2.txt). In the latter case, "
                                      "all coil channels are encoded across multiple columns in the text file. Use "
                                      "'gradient' to output the 1st order in the Gradient CS, otherwise, it outputs in "
                                      "the Shim CS.")
@click.option('--output-value-format', 'output_value_format', type=click.Choice(['delta', 'absolute']), default='delta',
              show_default=True,
              help="Coefficient values for the scanner coil. delta: Outputs the change of shim coefficients. "
                   "absolute: Outputs the absolute coefficient by taking into account the current shim settings. "
                   "This is effectively initial + shim. Scanner coil coefficients will be in the Shim coordinate "
                   "system unless the option --output-file-format is set to gradient. The delta value format should be "
                   "used in that case.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
@timeit
def dynamic(fname_fmap, fname_anat, fname_mask_anat, method, opt_criteria, slices, slice_factor, coils,
            dilation_kernel_size, scanner_coil_order, fname_sph_constr, fatsat, path_output, o_format_coil,
            o_format_sph, output_value_format, reg_factor, verbose):
    """ Static shim by fitting a fieldmap. Use the option --optimizer-method to change the shimming algorithm used to
    optimize. Use the options --slices and --slice-factor to change the shimming order/size of the slices.

    Example of use: st_b0shim dynamic --coil coil1.nii coil1_config.json --coil coil2.nii coil2_config.json
    --fmap fmap.nii --anat anat.nii --mask mask.nii --optimizer-method least_squares
    """

    scanner_coil_order = parse_orders(scanner_coil_order)
    # Set logger level
    set_all_loggers(verbose)

    # Load the fieldmap
    nii_fmap_orig = nib.load(fname_fmap)

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fmap_orig.get_fdata().ndim != 3:
        if nii_fmap_orig.get_fdata().ndim == 2:
            nii_fmap = nib.Nifti1Image(nii_fmap_orig.get_fdata()[..., np.newaxis], nii_fmap_orig.affine,
                                       header=nii_fmap_orig.header)
            nii_fmap = extend_fmap_to_kernel_size(nii_fmap, dilation_kernel_size, path_output)
        else:
            raise ValueError("Fieldmap must be 2d or 3d")
    else:
        # Extend the fieldmap if there are axes that have less voxels than the kernel size. This is done since we are
        # fitting a fieldmap to coil profiles and having a small number of voxels can lead to errors in fitting
        # (2 voxels in one dimension can differentiate order 1 at most), the parameter allows to have at least the
        # size of the kernel for each dimension This is usually useful in the through plane direction where we could
        # have less slices. To mitigate this, we create a 3d volume by replicating the slices on the edges.
        extending = False
        for i_axis in range(3):
            if nii_fmap_orig.shape[i_axis] < dilation_kernel_size:
                extending = True
                break

        if extending:
            nii_fmap = extend_fmap_to_kernel_size(nii_fmap_orig, dilation_kernel_size, path_output)
        else:
            nii_fmap = copy.deepcopy(nii_fmap_orig)

    # Prepare the output
    create_output_dir(path_output)

    # Load the anat
    nii_anat = nib.load(fname_anat)
    dim_info = nii_anat.header.get_dim_info()
    if dim_info[2] is None:
        logger.warning("The slice encoding direction is not specified in the NIfTI header, Shimming Toolbox will "
                       "assume it is in the third dimension.")
    else:
        if dim_info[2] != 2:
            # # Reorient nifti so that the slice is the last dim
            # anat = nii_anat.get_fdata()
            # # TODO: find index of dim_info
            # index_in = 0
            # index_out = 2
            #
            # # Swap axis in the array
            # anat = np.swapaxes(anat, index_in, index_out)
            #
            # # Affine must change
            # affine = copy.deepcopy(nii_anat.affine)
            # affine[:, index_in] = nii_anat.affine[:, index_out]
            # affine[:, index_out] = nii_anat.affine[:, index_in]
            # affine[index_out, 3] = nii_anat.affine[index_in, 3]
            # affine[index_in, 3] = nii_anat.affine[index_out, 3]
            #
            # nii_reorient = nib.Nifti1Image(anat, affine, header=nii_anat.header)
            # nib.save(nii_reorient, os.path.join(path_output, 'anat_reorient.nii.gz'))

            # Slice must be the 3rd dimension of the file
            # TODO: Reorient nifti so that the slice is the 3rd dim
            raise RuntimeError("Slice encode direction must be the 3rd dimension of the NIfTI file.")

    # Load anat json
    fname_anat_json = fname_anat.rsplit('.nii', 1)[0] + '.json'
    with open(fname_anat_json) as json_file:
        json_anat_data = json.load(json_file)

    # Load mask
    if fname_mask_anat is not None:
        nii_mask_anat = nib.load(fname_mask_anat)
    else:
        # If no mask is provided, shim the whole anat volume
        nii_mask_anat = nib.Nifti1Image(np.ones_like(nii_anat.get_fdata()), nii_anat.affine, header=nii_anat.header)

    if logger.level <= getattr(logging, 'DEBUG'):
        # Save inputs
        list_fname = [fname_fmap, fname_anat, fname_mask_anat]
        _save_nii_to_new_dir(list_fname, path_output)

    # Open json of the fmap
    fname_json = fname_fmap.split('.nii')[0] + '.json'
    # Read from json file
    if os.path.isfile(fname_json):
        with open(fname_json) as json_file:
            json_fm_data = json.load(json_file)
    else:
        raise OSError("Missing fieldmap json file")

    # Error out for unsupported inputs. If file format is in gradient CS, it must be 1st order and the output format be
    # delta. Only Siemens gradient coordinate system has been defined
    if o_format_sph == 'gradient':
        if output_value_format != 'delta':
            raise ValueError(f"Unsupported output value format: {output_value_format} for output file format: "
                             f"{o_format_sph}")
        if not (scanner_coil_order == [0, 1] or scanner_coil_order == [1]):
            raise ValueError(f"Unsupported scanner coil order: {scanner_coil_order} for output file format: "
                             f"{o_format_sph}")
        if json_fm_data.get('Manufacturer') != 'Siemens':
            raise NotImplementedError(f"Unsupported manufacturer: {json_fm_data.get('Manufacturer')} for output file"
                                      f"format: {o_format_sph}")

    # Read the current shim settings from the scanner
    scanner_shim_settings = ScannerShimSettings(json_fm_data)
    options = {'scanner_shim': scanner_shim_settings.shim_settings}

    # Load the coils
    list_coils = _load_coils(coils, scanner_coil_order, fname_sph_constr, nii_fmap, options['scanner_shim'],
                             json_fm_data.get('Manufacturer'), json_fm_data.get('ManufacturersModelName'))

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    if slices == 'auto':
        list_slices = parse_slices(fname_anat)
    else:
        list_slices = define_slices(n_slices, slice_factor, slices)
    logger.info(f"The slices to shim are:\n{list_slices}")
    # Get shimming coefficients
    # 1 ) Create the Shimming sequencer object
    sequencer = ShimSequencer(nii_fmap_orig, nii_anat, nii_mask_anat, list_slices, list_coils,
                              method=method,
                              opt_criteria=opt_criteria,
                              mask_dilation_kernel='sphere',
                              mask_dilation_kernel_size=dilation_kernel_size,
                              reg_factor=reg_factor,
                              path_output=path_output)
    # 2) Launch shim sequencer
    coefs = sequencer.shim()
    # Output
    # Load output options
    options['fatsat'] = _get_fatsat_option(json_anat_data, fatsat)

    list_fname_output = []
    end_channel = 0
    for i_coil, coil in enumerate(list_coils):

        # Figure out the start and end channels for a coil to be able to select it from the coefs
        n_channels = coil.dim[3]
        start_channel = end_channel
        end_channel = start_channel + n_channels

        # Select the coefficients for a coil
        coefs_coil = copy.deepcopy(coefs[:, start_channel:end_channel])

        # If it's a scanner
        if type(coil) == ScannerCoil:
            manufacturer = json_anat_data['Manufacturer']

            # If outputting in the gradient CS, it must be the 1st order, it must be in the delta CS and Siemens
            # The check has already been done earlier in the program to avoid processing and throw an error afterwards.
            # Therefore, we can only check for the o_format_sph.
            if o_format_sph == 'gradient':
                logger.debug("Converting Siemens scanner coil from Shim CS (LAI) to Gradient CS")
                # First convert to RAS
                orders = tuple([order for order in scanner_coil_order if order != 0])
                for i_shim in range(coefs.shape[0]):
                    # Convert coefficient
                    coefs_coil[i_shim, 1:] = shim_to_phys_cs(coefs_coil[i_shim, 1:], manufacturer, orders)

                # Convert coef of 1st order sph harmonics to Gradient coord system
                coefs_freq, coefs_phase, coefs_slice = phys_to_gradient_cs(coefs_coil[:, 1],
                                                                           coefs_coil[:, 2],
                                                                           coefs_coil[:, 3], fname_anat)

                coefs_coil[:, 1] = coefs_freq
                coefs_coil[:, 2] = coefs_phase
                coefs_coil[:, 3] = coefs_slice

            else:

                # If the output format is absolute, add the initial coefs
                if output_value_format == 'absolute':
                    initial_coefs = scanner_shim_settings.concatenate_shim_settings(scanner_coil_order)
                    for i_channel in range(n_channels):
                        # abs_coef = delta + initial
                        coefs_coil[:, i_channel] = coefs_coil[:, i_channel] + initial_coefs[i_channel]

                    list_fname_output += _save_to_text_file_static(coil, coefs_coil, list_slices, path_output,
                                                                   o_format_sph, options, coil_number=i_coil,
                                                                   default_coefs=initial_coefs)
                    continue

            list_fname_output += _save_to_text_file_static(coil, coefs_coil, list_slices, path_output, o_format_sph,
                                                           options, coil_number=i_coil)

        else:
            list_fname_output += _save_to_text_file_static(coil, coefs_coil, list_slices, path_output, o_format_coil,
                                                           options, coil_number=i_coil)

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")
    logger.info(f"Plotting figure(s)")
    sequencer.eval(coefs)
    logger.info(f" Plotting currents")

    # Plot the coefs after outputting the currents to the text file
    end_channel = 0
    for i_coil, coil in enumerate(list_coils):
        # Figure out the start and end channels for a coil to be able to select it from the coefs
        n_channels = coil.dim[3]
        start_channel = end_channel
        end_channel = start_channel + n_channels

        if type(coil) != ScannerCoil:
            # Select the coefficients for a coil
            coefs_coil = copy.deepcopy(coefs[:, start_channel:end_channel])
            # Plot a figure of the coefficients
            _plot_coefs(coil, list_slices, coefs_coil, path_output, i_coil,
                        bounds=[bound for bounds in coil.coef_channel_minmax.values() for bound in bounds])

    logger.info(f"Finished plotting figure(s)")


def _save_to_text_file_static(coil, coefs, list_slices, path_output, o_format, options, coil_number,
                              default_coefs=None):
    """o_format can either be 'slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil', 'gradient'"""

    n_channels = coil.dim[3]
    list_fname_output = []
    if o_format[-5:] == '-coil':

        fname_output = os.path.join(path_output, f"coefs_coil{coil_number}_{coil.name}.txt")
        with open(fname_output, 'w', encoding='utf-8') as f:
            # (len(slices) x n_channels)

            if o_format == 'chronological-coil':
                # Output per shim (chronological), output all channels for a particular shim, then repeat
                for i_shim in range(len(list_slices)):
                    # If fatsat pulse, set shim coefs to 0
                    if options['fatsat']:
                        for i_channel in range(n_channels):
                            if default_coefs is None:
                                # Output 0 (delta)
                                f.write(f"{0:.1f}, ")
                            else:
                                # Output initial coefs (absolute)
                                f.write(f"{default_coefs[i_channel]:.6f}, ")

                        f.write(f"\n")
                    for i_channel in range(n_channels):
                        f.write(f"{coefs[i_shim, i_channel]:.6f}, ")
                    f.write("\n")

            elif o_format == 'slicewise-coil':
                # Output per slice, output all channels for a particular slice, then repeat
                # Assumes all slices are in list_slices once which is the case for sequential, interleaved and
                # volume
                n_slices = np.sum([len(a_shim) for a_shim in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(a_shim) for a_shim in list_slices if i_slice in a_shim][0]
                    for i_channel in range(n_channels):
                        f.write(f"{coefs[i_shim, i_channel]:.6f}, ")
                    f.write("\n")

        list_fname_output.append(os.path.abspath(fname_output))

    elif o_format[-3:] == '-ch':

        # Write a file for each channel
        for i_channel in range(n_channels):
            fname_output = os.path.abspath(os.path.join(path_output,
                                                        f"coefs_coil{coil_number}_ch{i_channel}_{coil.name}.txt"))

            if o_format == 'chronological-ch':
                with open(fname_output, 'w', encoding='utf-8') as f:
                    # Each row will have one coef representing the shim in chronological order
                    for i_shim in range(len(list_slices)):
                        # If fatsat pulse, set shim coefs to 0
                        if options['fatsat']:
                            if default_coefs is None:
                                # Output 0 (delta)
                                f.write(f"{0:.1f},\n")
                            else:
                                # Output initial coefs (absolute)
                                f.write(f"{default_coefs[i_channel]:.6f},\n")
                        f.write(f"{coefs[i_shim, i_channel]:.6f},\n")

            if o_format == 'slicewise-ch':
                with open(fname_output, 'w', encoding='utf-8') as f:
                    # Each row will have one coef representing the shim in slicewise order
                    n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                    for i_slice in range(n_slices):
                        i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                        f.write(f"{coefs[i_shim, i_channel]:.6f}\n")

            list_fname_output.append(os.path.abspath(fname_output))
    else:  # o_format == 'gradient':

        for i_channel in range(n_channels):
            # Make sure there are 4 channels
            if n_channels != 4:
                raise RuntimeError("Gradient output format should only be used with 1st order scanner coils")

            name = {0: 'f0',
                    1: 'x',
                    2: 'y',
                    3: 'z'}

            fname_output = os.path.join(path_output, f"{name[i_channel]}shim_gradients.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]

                    if i_channel == 0:
                        # f0, Output is in Hz
                        f.write(f"corr_vec[0][{i_slice}]= "
                                f"{coefs[i_shim, i_channel]:.6f}\n")
                    else:
                        # For Gx, Gy, Gz: Divide by 1000 for mT/m
                        f.write(f"corr_vec[0][{i_slice}]= "
                                f"{coefs[i_shim, i_channel] / 1000:.6f}\n")

                    # Static shimming does not have a a riro component
                    f.write(f"corr_vec[1][{i_slice}]= "
                            f"{0:.12f}\n")
                    # Arbitrarily chose a mean pressure of 2000 to satisfy the sequence
                    f.write(f"corr_vec[2][{i_slice}]= {2000:.3f}\n")

            list_fname_output.append(os.path.abspath(fname_output))

    return list_fname_output


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils_static', nargs=2, multiple=True, type=(click.Path(exists=True), click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__dir_config_scanner_constraints__}")
@click.option('--coil-riro', 'coils_riro', nargs=2, multiple=True,
              type=(click.Path(exists=True), click.Path(exists=True)), required=False,
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If this option is used, these coil profiles will be used for "
                   "the RIRO optimization, otherwise, the coils from the --coil options will be used."
                   "If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__dir_config_scanner_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Timeseries of B0 fieldmap.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Anatomical image to apply the correction onto.")
@click.option('--resp', 'fname_resp', type=click.Path(exists=True), required=True,
              help="Siemens respiratory file containing pressure data.")
@click.option('--mask-static', 'fname_mask_anat_static', type=click.Path(exists=True), required=False,
              help="Mask defining the static spatial region to shim.")
@click.option('--mask-riro', 'fname_mask_anat_riro', type=click.Path(exists=True), required=False,
              help="Mask defining the time varying (i.e. RIRO, Respiration-Induced Resonance Offset) "
                   "region to shim.")
@click.option('--scanner-coil-order', 'scanner_coil_order_static', type=click.STRING, default='-1', show_default=True,
              help="Spherical harmonics orders to be used in static optimization. "
                   f"Available orders: {AVAILABLE_ORDERS}. "
                   "Orders should be writen with a coma separating the values. (i.e. 0,1,2)"
                   "The 0th order is the f0 frequency.")
@click.option('--scanner-coil-order-riro', 'scanner_coil_order_riro', type=click.STRING, default=None,
              show_default=True,
              help="Spherical harmonics orders to be used in RIRO optimization. If not set, the same orders as "
                   "--scanner-coil-order will be used for RIRO"
                   f"Available orders: {AVAILABLE_ORDERS}. "
                   "Orders should be writen with a coma separating the values. (i.e. 0,1,2)"
                   "The 0th order is the f0 frequency.")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(), default="",
              help=f"Constraints for the scanner coil. Example file located: {__dir_config_scanner_constraints__}")
@click.option('--slices', type=click.Choice(['interleaved', 'sequential', 'volume', 'auto']), required=False,
              default='auto', show_default=True,
              help="Define the slice ordering. If set to 'auto', automatically parse the target image.")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shimmed group. Used when '--slices' is not set to 'auto'. For example, if the "
                   "'--slice-factor' value is '3', then with the 'sequential' mode, shimming will be performed "
                   "independently on the following groups: {0,1,2}, {3,4,5}, etc. With the mode 'interleaved', "
                   "it will be: {0,2,4}, {1,3,5}, etc.")
@click.option('--optimizer-method', 'method', type=click.Choice(['least_squares', 'pseudo_inverse',
                                                                 'quad_prog']), required=False,
              default='quad_prog', show_default=True,
              help="Method used by the optimizer. LS and QP will respect the constraints,"
                   "PS will not respect the constraints")
@click.option('--optimizer-criteria', 'opt_criteria', type=click.Choice(['mse', 'mae']), required=False,
              default='mse', show_default=True,
              help="Criteria of optimization for the optimizer 'least_squares'."
                   " mse: Mean Squared Error, mae: Mean Absolute Error")
@click.option('--regularization-factor', 'reg_factor', type=click.FLOAT, required=False, default=0.0, show_default=True,
              help="Regularization factor for the current when optimizing. A higher coefficient will penalize higher "
                   "current values while 0 provides no regularization. Not relevant for 'pseudo-inverse' "
                   "optimizer_method.")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False, default='3',
              show_default=True,
              help="Number of voxels to consider outside of the masked area. For example, when doing dynamic shimming "
                   "with a linear gradient, the coefficient corresponding to the gradient orthogonal to a single "
                   "slice cannot be estimated: there must be at least 2 (ideally 3) points to properly estimate the "
                   "linear term. When using 2nd order or more, more dilation is necessary.")
@click.option('--fatsat', type=click.Choice(['auto', 'yes', 'no']), default='auto', show_default=True,
              help="Describe what to do with a fat saturation pulse. 'auto': It will parse the NIfTI file "
                   "for a fat-sat pulse and add shim coefficients of 0s before every shim group when using "
                   "'chronological-...' output-file-format-coil. 'no': It will not add 0s. 'yes': It will add 0s.")
@click.option('-o', '--output', 'path_output', type=click.Path(), default=os.path.abspath(os.curdir),
              show_default=True, help="Directory to output coil text file(s).")
@click.option('--output-file-format-coil', 'o_format_coil',
              type=click.Choice(['slicewise-ch', 'chronological-ch']), default='slicewise-ch', show_default=True,
              help="Syntax used to describe the sequence of shim events. "
                   "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                   "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                   "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                   "captured by the controller of the shim amplifier. For both 'slicewice' and 'chronological', "
                   "there will be one output file per coil channel (coil1_ch1.txt, coil1_ch2.txt, etc.). The static, "
                   "time-varying and mean pressure are encoded in the columns of each file.")
@click.option('--output-file-format-scanner', 'o_format_sph',
              type=click.Choice(['slicewise-ch', 'chronological-ch', 'gradient']), default='slicewise-ch',
              show_default=True,
              help="Syntax used to describe the sequence of shim events. "
                   "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                   "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                   "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                   "captured by the controller of the shim amplifier. In both cases, there will be one output "
                   "file per coil channel (coil1_ch1.txt, coil1_ch2.txt, etc.). The static, "
                   "time-varying and mean pressure are encoded in the columns of each file. Use "
                   "'gradient' to output the scanner 1st order in the Gradient CS, otherwise, it outputs "
                   "in the Shim CS.")
@click.option('--output-value-format', 'output_value_format', type=click.Choice(['delta', 'absolute']),
              default='delta', show_default=True,
              help="Coefficient values for the scanner coil. delta: Outputs the change of shim coefficients. "
                   "absolute: Outputs the absolute coefficient by taking into account the current shim settings. "
                   "This is effectively initial + shim. Scanner coil coefficients will be in the Shim coordinate "
                   "system unless the option --output-file-format is set to gradient. The delta value format should be "
                   "used in that case.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
@timeit
def realtime_dynamic(fname_fmap, fname_anat, fname_mask_anat_static, fname_mask_anat_riro, fname_resp, method,
                     opt_criteria, slices, slice_factor, coils_static, coils_riro, dilation_kernel_size,
                     scanner_coil_order_static, scanner_coil_order_riro, fname_sph_constr, fatsat, path_output,
                     o_format_coil, o_format_sph, output_value_format, reg_factor, verbose):
    """ Realtime shim by fitting a fieldmap to a pressure monitoring unit. Use the option --optimizer-method to change
    the shimming algorithm used to optimize. Use the options --slices and --slice-factor to change the shimming
    order/size of the slices.

    Example of use: st_b0shim realtime-dynamic --coil coil1.nii coil1_config.json --coil coil2.nii coil2_config.json
    --fmap fmap.nii --anat anat.nii --mask-static mask.nii --resp trace.resp --optimizer-method least_squares
    """
    # Set coils and scanner order for riro if none were indicated
    if scanner_coil_order_riro is None:
        scanner_coil_order_riro = scanner_coil_order_static

    scanner_coil_order_static = parse_orders(scanner_coil_order_static)
    scanner_coil_order_riro = parse_orders(scanner_coil_order_riro)

    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(path_output)

    # Load the fieldmap
    nii_fmap_orig = nib.load(fname_fmap)

    # Make sure the fieldmap has the appropriate dimensions
    if nii_fmap_orig.get_fdata().ndim != 4:
        raise ValueError("Fieldmap must be 4d (dim1, dim2, dim3, t)")

    # Extend the fieldmap if there are axes that have less voxels than the kernel size. This is done since we are
    # fitting a fieldmap to coil profiles and having a small number of voxels can lead to errors in fitting (2 voxels
    # in one dimension can differentiate order 1 at most), the parameter allows to have at least the size of the kernel
    # for each dimension This is usually useful in the through plane direction where we could have less slices.
    # To mitigate this, we create a 3d volume by replicating the slices on the edges.
    extending = False
    for i_axis in range(3):
        if nii_fmap_orig.shape[i_axis] < dilation_kernel_size:
            extending = True
            break

    if extending:
        nii_fmap = extend_fmap_to_kernel_size(nii_fmap_orig, dilation_kernel_size, path_output)
    else:
        nii_fmap = copy.deepcopy(nii_fmap_orig)

    # Load the anat
    nii_anat = nib.load(fname_anat)
    dim_info = nii_anat.header.get_dim_info()
    if dim_info[2] != 2:
        # Slice must be the 3rd dimension of the file
        # TODO: Reorient nifti so that the slice is the 3rd dim
        raise RuntimeError("Slice encode direction must be the 3rd dimension of the NIfTI file.")

    # Load anat json
    fname_anat_json = fname_anat.rsplit('.nii', 1)[0] + '.json'
    with open(fname_anat_json) as json_file:
        json_anat_data = json.load(json_file)

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

    # Open json of the fmap
    fname_json = fname_fmap.split('.nii')[0] + '.json'
    # Read from json file
    if os.path.isfile(fname_json):
        with open(fname_json) as json_file:
            json_fm_data = json.load(json_file)
    else:
        raise OSError("Missing fieldmap json file")

    # Error out for unsupported inputs. If file format is in gradient CS, it must be 1st order and the output format be
    # delta.
    if o_format_sph == 'gradient':
        if output_value_format == 'absolute':
            raise ValueError(f"Unsupported output value format: {output_value_format} for output file format: "
                             f"{o_format_sph}")
        if not (scanner_coil_order_static == [0, 1] or scanner_coil_order_static == [1]) or \
                not (scanner_coil_order_riro == [0, 1] or scanner_coil_order_riro == [1]):
            raise ValueError(f"Unsupported scanner coil order: {scanner_coil_order_static} for output file format: "
                             f"{o_format_sph}")
        if json_fm_data['Manufacturer'] != 'Siemens':
            raise ValueError(f"Unsupported manufacturer: {json_fm_data['manufacturer']} for output file format: "
                             f"{o_format_sph}")

    # Read the current shim settings from the scanner
    scanner_shim_settings = ScannerShimSettings(json_fm_data)
    options = {'scanner_shim': scanner_shim_settings.shim_settings}

    # Load the coils
    list_coils_static = _load_coils(coils_static, scanner_coil_order_static, fname_sph_constr, nii_fmap,
                                    options['scanner_shim'], json_fm_data['Manufacturer'],
                                    json_fm_data['ManufacturersModelName'])
    list_coils_riro = _load_coils(coils_riro, scanner_coil_order_riro, fname_sph_constr, nii_fmap,
                                  options['scanner_shim'], json_fm_data['Manufacturer'],
                                  json_fm_data['ManufacturersModelName'])

    if logger.level <= getattr(logging, 'DEBUG'):
        # Save inputs
        list_fname = [fname_fmap, fname_anat, fname_mask_anat_static, fname_mask_anat_riro]
        _save_nii_to_new_dir(list_fname, path_output)

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    if slices == 'auto':
        list_slices = parse_slices(fname_anat)
    else:
        list_slices = define_slices(n_slices, slice_factor, slices)
    logger.info(f"The slices to shim are: {list_slices}")

    # Load PMU
    pmu = PmuResp(fname_resp)
    # 1 ) Create the real time pmu sequencer object
    sequencer = RealTimeSequencer(nii_fmap_orig, json_fm_data, nii_anat, nii_mask_anat_static,
                                  nii_mask_anat_riro,
                                  list_slices, pmu, list_coils_static, list_coils_riro,
                                  method=method,
                                  opt_criteria=opt_criteria,
                                  mask_dilation_kernel='sphere',
                                  mask_dilation_kernel_size=dilation_kernel_size,
                                  reg_factor=reg_factor,
                                  path_output=path_output)
    # 2) Launch the sequencer
    out = sequencer.shim()
    coefs_static, coefs_riro, mean_p, p_rms = out

    # Output
    # Load output options
    options['fatsat'] = _get_fatsat_option(json_anat_data, fatsat)

    # Get common coils between static and riro // Comparison based on coil name
    coil_static_only = [coil for coil in list_coils_static if coil not in list_coils_riro]
    coil_riro_only = [coil for coil in list_coils_riro if coil not in list_coils_static]
    list_coils_common = [coil for coil in list_coils_static if coil in list_coils_riro]
    # Create a list of all coils used in optimization
    all_coils = list_coils_common + coil_static_only + coil_riro_only

    index = 0
    coil_indexes_static = {}
    for coil in list_coils_static:
        if type(coil) == Coil:
            coil_indexes_static[coil.name] = [index, index + len(coil.coef_channel_minmax['coil'])]
            index += len(coil.coef_channel_minmax['coil'])
        else:
            coil_indexes_static[coil.name] = {}
            for key in coil.coef_channel_minmax:
                coil_indexes_static[coil.name][key] = [index, index + len(coil.coef_channel_minmax[key])]
                index += len(coil.coef_channel_minmax[key])

    index = 0
    coil_indexes_riro = {}
    for coil in list_coils_riro:
        if type(coil) == Coil:
            coil_indexes_riro[coil.name] = [index, index + len(coil.coef_channel_minmax['coil'])]
            index += len(coil.coef_channel_minmax['coil'])
        else:
            coil_indexes_riro[coil.name] = {}
            for key in coil.coef_channel_minmax:
                coil_indexes_riro[coil.name][key] = [index, index + len(coil.coef_channel_minmax[key])]
                index += len(coil.coef_channel_minmax[key])

    list_fname_output = []
    for i_coil, coil in enumerate(all_coils):
        # Figure out the start and end channels for a coil to be able to select it from the coefs

        # If it's a scanner
        if type(coil) == ScannerCoil:
            if coil in list_coils_common:
                keys = [str(order) for order in AVAILABLE_ORDERS
                        if (order != -1 and (str(order) in coil_indexes_riro[coil.name]
                                             or str(order) in coil_indexes_static[coil.name]))]
            elif coil in coil_static_only:
                keys = [str(order) for order in AVAILABLE_ORDERS
                        if (order != -1 and str(order) in coil_indexes_static[coil.name])]
            elif coil in coil_riro_only:
                keys = [str(order) for order in AVAILABLE_ORDERS
                        if (order != -1 and str(order) in coil_indexes_riro[coil.name])]

            for key in keys:
                if coil in list_coils_riro:
                    if key in coil_indexes_riro[coil.name]:
                        coefs_coil_riro = copy.deepcopy(
                            coefs_riro[:, coil_indexes_riro[coil.name][key][0]:coil_indexes_riro[coil.name][key][1]])
                    else:
                        coefs_coil_riro = np.zeros_like(coefs_static[:, coil_indexes_static[coil.name][key][0]:
                                                                     coil_indexes_static[coil.name][key][1]])
                else:
                    coefs_coil_riro = np.zeros_like(
                        coefs_static[:, coil_indexes_static[coil.name][key][0]:coil_indexes_static[coil.name][key][1]])

                if coil in list_coils_static:
                    if key in coil_indexes_static[coil.name]:
                        coefs_coil_static = copy.deepcopy(coefs_static[:, coil_indexes_static[coil.name][key][0]:
                                                                       coil_indexes_static[coil.name][key][1]])
                    else:
                        coefs_coil_static = np.zeros_like(coefs_coil_riro)
                else:
                    coefs_coil_static = np.zeros_like(coefs_coil_riro)

                manufacturer = json_anat_data['Manufacturer']
                # If outputting in the gradient CS, it must be the 1st order and must be in the delta CS and Siemens
                # The check has already been done earlier in the program to avoid processing and throw an error
                # afterwards.
                # Therefore, we can only check for the o_format_sph.
                if o_format_sph == 'gradient':
                    if key == '0':
                        save_coefs_static = coefs_coil_static
                        if coefs_coil_riro is not None:
                            save_coefs_riro = coefs_coil_riro
                        else:
                            save_coefs_riro = None
                        has0 = True
                        continue
                    elif key == '1' and has0:
                        save_coefs_static = np.concatenate((save_coefs_static, coefs_coil_static), axis=1)
                        if save_coefs_riro is not None:
                            save_coefs_riro = np.concatenate((save_coefs_riro, coefs_coil_riro), axis=1)
                        elif coefs_coil_riro is not None:
                            save_coefs_riro = coefs_coil_riro
                        else:
                            raise ValueError("Orders do not match gradient")
                    coefs_coil_static = save_coefs_static
                    coefs_coil_riro = save_coefs_riro
                    logger.debug("Converting scanner coil from Shim CS to Gradient CS")
                    orders_static = tuple([order for order in scanner_coil_order_static if order != 0])
                    orders_riro = tuple([order for order in scanner_coil_order_riro if order != 0])
                    # First convert coefficients from Shim CS to RAS
                    for i_shim in range(coefs_coil_static.shape[0]):
                        # Convert coefficient
                        coefs_coil_static[i_shim, 1:] = shim_to_phys_cs(coefs_coil_static[i_shim, 1:], manufacturer,
                                                                        orders_static)
                        coefs_coil_riro[i_shim, 1:] = shim_to_phys_cs(coefs_coil_riro[i_shim, 1:], manufacturer,
                                                                      orders_riro)

                    # RAS to gradient
                    coefs_st_freq, coefs_st_phase, coefs_st_slice = phys_to_gradient_cs(
                        coefs_coil_static[:, 1],
                        coefs_coil_static[:, 2],
                        coefs_coil_static[:, 3],
                        fname_anat)
                    coefs_coil_static[:, 1] = coefs_st_freq
                    coefs_coil_static[:, 2] = coefs_st_phase
                    coefs_coil_static[:, 3] = coefs_st_slice

                    coefs_riro_freq, coefs_riro_phase, coefs_riro_slice = phys_to_gradient_cs(
                        coefs_coil_riro[:, 1],
                        coefs_coil_riro[:, 2],
                        coefs_coil_riro[:, 3],
                        fname_anat)
                    coefs_coil_riro[:, 1] = coefs_riro_freq
                    coefs_coil_riro[:, 2] = coefs_riro_phase
                    coefs_coil_riro[:, 3] = coefs_riro_slice

                else:

                    # If the output format is absolute, add the initial coefs
                    if output_value_format == 'absolute' and coefs_coil_static is not None:
                        initial_coefs = scanner_shim_settings.concatenate_shim_settings(scanner_coil_order_static)
                        for i_channel in range(coefs_coil_static.shape[-1]):
                            # abs_coef = delta + initial
                            coefs_coil_static[:, i_channel] = coefs_coil_static[:, i_channel] + initial_coefs[i_channel]
                            # riro does not change

                            list_fname_output += _save_to_text_file_rt(coil, coefs_coil_static, coefs_coil_riro, mean_p,
                                                                       list_slices, path_output, o_format_sph, options,
                                                                       i_coil, int(key) ** 2,
                                                                       default_st_coefs=initial_coefs)
                        continue

                list_fname_output += _save_to_text_file_rt(coil, coefs_coil_static, coefs_coil_riro, mean_p,
                                                           list_slices,
                                                           path_output, o_format_sph, options, i_coil, int(key) ** 2)

        else:  # Custom coil
            if coil in list_coils_riro:
                coefs_coil_riro = copy.deepcopy(
                    coefs_riro[:, coil_indexes_riro[coil.name][0]:coil_indexes_riro[coil.name][1]])
            else:
                coefs_coil_riro = np.zeros_like(
                    coefs_static[:, coil_indexes_static[coil.name][0]:coil_indexes_static[coil.name][1]])
            if coil in list_coils_static:
                coefs_coil_static = copy.deepcopy(
                    coefs_static[:, coil_indexes_static[coil.name][0]:coil_indexes_static[coil.name][1]])
            else:
                coefs_coil_static = np.zeros_like(coefs_coil_riro)

            list_fname_output += _save_to_text_file_rt(coil, coefs_coil_static, coefs_coil_riro, mean_p, list_slices,
                                                       path_output, o_format_coil, options, i_coil, 0)

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")
    logger.info(f"Plotting figure(s)")
    sequencer.eval(coefs_static, coefs_riro, mean_p, p_rms)
    logger.info(f"Plotting Currents")
    # Plot the coefs after outputting the currents to the text file
    end_channel = 0

    for i_coil, coil in enumerate(all_coils):
        # Figure out the start and end channels for a coil to be able to select it from the coefs
        if type(coil) != ScannerCoil:
            if coil in list_coils_riro:
                coefs_coil_riro = copy.deepcopy(
                    coefs_riro[:, coil_indexes_riro[coil.name][0]:coil_indexes_riro[coil.name][1]])
            else:
                coefs_coil_riro = None
            if coil in list_coils_static:
                coefs_coil_static = copy.deepcopy(
                    coefs_static[:, coil_indexes_static[coil.name][0]:coil_indexes_static[coil.name][1]])
            else:
                coefs_coil_static = np.zeros_like(coefs_coil_riro)
            # Plot a figure of the coefficients
            _plot_coefs(coil, list_slices, coefs_coil_static, path_output, i_coil, coefs_coil_riro,
                        pres_probe_max=pmu.max - mean_p, pres_probe_min=pmu.min - mean_p,
                        bounds=[bound for bounds in coil.coef_channel_minmax.values() for bound in bounds])

    logger.info(f"Finished plotting figure(s)")


def _save_to_text_file_rt(coil, currents_static, currents_riro, mean_p, list_slices, path_output, o_format,
                          options, coil_number, channel_start, default_st_coefs=None):
    """o_format can either be 'chronological-ch', 'chronological-coil', 'gradient'"""

    list_fname_output = []
    if currents_riro is not None:
        n_channels = currents_riro.shape[-1]
    else:
        n_channels = currents_static.shape[-1]
    # Write a file for each channel
    for i_channel in range(n_channels):

        if o_format == 'chronological-ch':
            fname_output = os.path.join(path_output,
                                        f"coefs_coil{coil_number}_ch{channel_start + i_channel}_{coil.name}.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                # Each row will have 3 coef representing the static, riro and mean_p in chronological order
                for i_shim in range(len(list_slices)):
                    # If fatsat pulse, set shim coefs to 0 and output mean pressure
                    if options['fatsat']:
                        if default_st_coefs is None:
                            # Output 0 (delta)
                            f.write(f"{0:.1f}, {0:.1f}, {mean_p:.4f},\n")
                        else:
                            # Output initial coefs (absolute)
                            f.write(f"{default_st_coefs[i_channel]:.1f}, {0:.1f}, {mean_p:.4f},\n")
                    if currents_static is not None:
                        f.write(f"{currents_static[i_shim, i_channel]:.6f}, ")
                    if currents_riro is not None:
                        f.write(f"{currents_riro[i_shim, i_channel]:.12f}, ")
                    f.write(f"{mean_p:.4f},\n")

        elif o_format == 'slicewise-ch':
            fname_output = os.path.join(path_output,
                                        f"coefs_coil{coil_number}_ch{channel_start + i_channel}_{coil.name}.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                # Each row will have one coef representing the static, riro and mean_p in slicewise order
                n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]

                    if currents_static is not None:
                        f.write(f"{currents_static[i_shim, i_channel]:.6f}, ")
                    if currents_riro is not None:
                        f.write(f"{currents_riro[i_shim, i_channel]:.12f}, ")
                    f.write(f"{mean_p:.4f},\n")

        else:  # o_format == 'gradient':

            # Make sure there are 4 channels
            if n_channels != 4:
                raise RuntimeError("Gradient output format should only be used with 1st order scanner coils")

            name = {0: 'f0',
                    1: 'x',
                    2: 'y',
                    3: 'z'}

            fname_output = os.path.join(path_output, f"{name[i_channel]}shim_gradients.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]

                    if i_channel == 0:
                        # f0, Output is in Hz
                        if currents_static is not None:
                            f.write(f"corr_vec[0][{i_slice}]= "
                                    f"{currents_static[i_shim, i_channel]:.6f}\n")
                        if currents_riro is not None:
                            f.write(f"corr_vec[1][{i_slice}]= "
                                    f"{currents_riro[i_shim, i_channel]:.12f}\n")
                        f.write(f"corr_vec[2][{i_slice}]= {mean_p:.3f}\n")

                    else:
                        # For Gx, Gy, Gz: Divide by 1000 for mT/m
                        if currents_static is not None:
                            f.write(f"corr_vec[0][{i_slice}]= "
                                    f"{currents_static[i_shim, i_channel] / 1000:.6f}\n")
                        if currents_riro is not None:
                            f.write(f"corr_vec[1][{i_slice}]= "
                                    f"{currents_riro[i_shim, i_channel] / 1000:.12f}\n")
                        f.write(f"corr_vec[2][{i_slice}]= {mean_p:.3f}\n")

        list_fname_output.append(os.path.abspath(fname_output))

    return list_fname_output


def parse_orders(orders: str):
    orders = orders.split(',')
    try:
        orders = [int(order) for order in orders]
        orders.sort()
        if any(order not in AVAILABLE_ORDERS for order in orders):
            raise ValueError(f'Orders must be selected from: {AVAILABLE_ORDERS}')
        return orders
    except ValueError:
        raise ValueError(f"Invalid orders: {orders}\n Orders must be integers ")


def _load_coils(coils, orders, fname_constraints, nii_fmap, scanner_shim_settings, manufacturer,
                manufacturers_model_name):
    # ! Modify description if everything works
    """ Loads the Coil objects from filenames

    Args:
        coils (list): List of tuples(fname_nii, fname_json) of coil profiles and constraints
        orders (list): Orders of the scanner coils (0 or 1 or 2)
        fname_constraints (str): Filename of the constraints of the scanner coils
        nii_fmap (nib.Nifti1Image): Nibabel object of the fieldmap
        scanner_shim_settings (dict): Dictionary containing the shim settings of the scanner ('0', '1', '2')
        manufacturer (str): Name of the MRI manufacturer
        manufacturers_model_name (str): Name of the scanner

    Returns:
        list: List of Coil objects containing the custom coils followed by the scanner coil if requested
    """
    list_coils = []

    # Load custom coils
    for coil in coils:
        nii_coil_profiles = nib.load(coil[0])
        with open(coil[1]) as json_file:
            constraints = json.load(json_file)
        list_coils.append(Coil(nii_coil_profiles.get_fdata(), nii_coil_profiles.affine, constraints))

    if len(list_coils) != len(set(list_coils)):
        raise ValueError("Coils must be unique. Make sure different coils have different names.")

    # Create the spherical harmonic coil profiles of the scanner
    if -1 not in orders:

        if os.path.isfile(fname_constraints):
            with open(fname_constraints) as json_file:
                sph_contraints = json.load(json_file)
            orders_to_delete = []
            for key in sph_contraints['coef_channel_minmax']:
                if key not in str(orders):
                    orders_to_delete.append(key)
            for key in orders_to_delete:
                del sph_contraints['coef_channel_minmax'][key]
        else:
            sph_contraints = get_scanner_constraints(manufacturers_model_name, orders)

        sph_contraints_calc = calculate_scanner_constraints(sph_contraints, scanner_shim_settings, orders)
        scanner_coil = ScannerCoil(nii_fmap.shape[:3], nii_fmap.affine, sph_contraints_calc, orders,
                                   manufacturer=manufacturer)
        list_coils.append(scanner_coil)

    # Make sure a coil is selected
    if len(list_coils) == 0:
        raise RuntimeError("No custom or scanner coils were selected. Use --coil and/or --scanner-coil-order")

    return list_coils


def calculate_scanner_constraints(constraints: dict, scanner_shim_settings, orders):
    # ! Modify description if everything works
    """ Calculate the constraints that should be used for the scanner by considering the current shim settings and the
        absolute bounds

    Args:
        constraints (dict): Constraints of the scanner coils
        scanner_shim_settings (dict): Dictionary containing the shim settings of the scanner ('0', '1', '2')
        orders (list): Order of the scanner coils (0 or 1 or 2)
        manufacturer (str): Name of the MRI manufacturer

    Returns:
        dict: Updated constraints of the scanner
    """

    def _initial_in_bounds(coefs: dict, bounds: dict):
        """Makes sure the initial values are within the bounds of the constraints"""
        if coefs.keys() != bounds.keys():
            raise RuntimeError("The scanner coil's orders is not the same length as the initial orders")
        if any(len(coefs[key]) != len(bounds[key]) for key in bounds):
            raise RuntimeError("The scanner coil's bounds is not the same length as the initial bounds")

        for key in coefs:
            for (bound, coef) in zip(bounds[key], coefs[key]):
                if bound[0] is None and bound[1] is None:
                    continue
                if bound[1] is not None:
                    if not coef <= bound[1]:
                        logger.warning(f"Initial scanner coefs are outside the bounds allowed in the constraints: "
                                       f"{bound}, initial: {coef}")
                if bound[0] is not None:
                    if not bound[0] <= coef:
                        logger.warning(f"Initial scanner coefs are outside the bounds allowed in the constraints: "
                                       f"{bound}, initial: {coef}")

    # Set the initial coefficients to 0
    initial_coefs = {}
    for order in orders:
        initial_coefs[str(order)] = [0] * (order * 2 + 1)
    if initial_coefs == {}:
        initial_coefs = None

    # Restrict the constraints to the provided order
    constraints['coef_channel_minmax'] = restrict_sph_constraints(constraints['coef_channel_minmax'], orders)

    # If the scanner coefficients are valid, update the initial coefficients
    if scanner_shim_settings['has_valid_settings']:
        if scanner_shim_settings['0'] is not None and 0 in orders:
            initial_coefs['0'] = np.array(scanner_shim_settings['0'])
        if scanner_shim_settings['1'] is not None and 1 in orders:
            initial_coefs['1'] = scanner_shim_settings['1']
        if scanner_shim_settings['2'] is not None and 2 in orders:
            initial_coefs['2'] = scanner_shim_settings['2']

        # Make sure the initial coefficients are within the specified bounds
        _initial_in_bounds(initial_coefs, constraints['coef_channel_minmax'])

    # Update the bounds to what they should be by taking into account that the fieldmap was acquired using some
    # shimming
    constraints['coef_channel_minmax'] = new_bounds_from_currents(initial_coefs,
                                                                  constraints['coef_channel_minmax']
                                                                  )
    return constraints


def _save_nii_to_new_dir(list_fname, path_output):
    """List of nii to save to a new output folder"""
    logger.debug(f"Saving CLI inputs to: {path_output}")
    for fname in list_fname:
        if fname is None:
            continue
        nii = nib.load(fname)
        fname_to_save = os.path.join(path_output, os.path.basename(fname))
        nib.save(nii, fname_to_save)


def _get_fatsat_option(json_anat, fatsat):
    """ Return if the fat saturation option should be turned on or off. This function mainly exists to resolve the 'auto'
        case

    Args:
        json_anat (dict): BIDS Json sidecar
        fatsat (str): String containing either : 'yes', 'no' or 'auto'

    Returns:
        bool: Whether to activate fatsat or not
    """
    fatsat_option = False

    if fatsat == 'auto':
        if 'ScanOptions' in json_anat:
            if 'FS' in json_anat['ScanOptions']:
                logger.debug("Fat Saturation pulse detected")
                fatsat_option = True
    elif fatsat == 'yes':
        fatsat_option = True

    return fatsat_option


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


@timeit
def _plot_coefs(coil, slices, static_coefs, path_output, coil_number, rt_coefs=None, pres_probe_min=None,
                pres_probe_max=None, units='', bounds=None):
    # Find which slices are not shimmed and group them (smaller file size and reduce the plot saving time)
    shimmed_slice_index = []
    n_shims = len(slices)
    slices_index_wo_shim = []
    unused_slice = False
    for i_shim in range(n_shims):
        # Static case
        if np.any(static_coefs[i_shim]):
            shimmed_slice_index.append(i_shim)
            continue

        # Realtime case
        if rt_coefs is not None:
            if np.any(rt_coefs[i_shim]):
                shimmed_slice_index.append(i_shim)
                continue

        # Get a string with the number of all the unshimmed slices
        slices_index_wo_shim.append(i_shim)
        unused_slice = True

    # Find min and max values of the plots
    # Calculate the min and max of the bounds if it's an input
    if bounds is not None:
        bounds = np.array(bounds)
        min_y = bounds.min()
        max_y = bounds.max()
    else:
        min_y = None
        max_y = None

    # Calculate the min and max coefficient for the combined static + riro * (acq_pressure - mean_p)
    # It can expand the min/max of the bounds if necessary
    if rt_coefs is not None:
        for i_shim in range(n_shims):
            n_channels = static_coefs.shape[1]
            for i_channel in range(n_channels):
                coef = rt_coefs[i_shim, i_channel]
                if coef > 0:
                    temp_min = static_coefs[i_shim, i_channel] + coef * pres_probe_min
                    temp_max = static_coefs[i_shim, i_channel] + coef * pres_probe_max
                else:
                    temp_min = static_coefs[i_shim, i_channel] + coef * pres_probe_max
                    temp_max = static_coefs[i_shim, i_channel] + coef * pres_probe_min

                if min_y is None or min_y > temp_min:
                    min_y = temp_min
                if max_y is None or max_y < temp_max:
                    max_y = temp_max

    # If its static optimization, find the min and max. It can expand the bounds.
    else:
        temp_min = np.array(static_coefs).min()
        if min_y is None or min_y > temp_min:
            min_y = temp_min
        temp_max = np.array(static_coefs).max()
        if max_y is None or max_y < temp_max:
            max_y = np.array(static_coefs).max()

    # Plot the currents
    n_plots = len(shimmed_slice_index)
    if unused_slice:
        n_plots += 1

    fig = Figure(figsize=(8, 4 * n_plots), tight_layout=True)
    for i_plot, slice_index in enumerate(shimmed_slice_index):

        if rt_coefs is not None:
            rt_coef_tmp = rt_coefs[slice_index]
        else:
            rt_coef_tmp = None

        _add_sub_figure(fig, i_plot + 1, n_plots, static_coefs[slice_index], bounds, min_y, max_y, units,
                        slices[slice_index], rt_coef_tmp, pres_probe_min, pres_probe_max)

    # Add a subplot for all the non shimmed slices
    if unused_slice:
        i_unshimmed_slice = slices_index_wo_shim[0]
        slices_wo_shim = tuple(j for i in slices_index_wo_shim for j in slices[i])
        _add_sub_figure(fig, n_plots, n_plots, static_coefs[i_unshimmed_slice], bounds,
                        min_y, max_y, units, slices_wo_shim)

    # Save the figure
    fname_figure = os.path.join(path_output, f"fig_currents_per_slice_group_coil{coil_number}_{coil.name}.png")
    fig.savefig(fname_figure, bbox_inches='tight')
    logger.debug(f"Saved figure: {fname_figure}")


def _add_sub_figure(fig, i_plot, n_plots, static_coefs, bounds, min_y, max_y, units, slice_number, rt_coefs=None,
                    pres_probe_min=None, pres_probe_max=None):
    # Make a subplot for slices
    # If it's the recap subplot for all the slices where the correction is null then we need to take an index further to
    # not have visual problem

    ax = fig.add_subplot(n_plots, 1, i_plot)
    n_channels = len(static_coefs)

    # Add realtime component as an errorbar
    if rt_coefs is not None:
        riro = np.zeros((2, rt_coefs.shape[0]))
        for i_slice in range(rt_coefs.shape[0]):
            riro[0, i_slice] = np.abs(min(rt_coefs[i_slice] * -pres_probe_min, rt_coefs[i_slice] * pres_probe_max))
            riro[1, i_slice] = np.abs(max(rt_coefs[i_slice] * -pres_probe_min, rt_coefs[i_slice] * pres_probe_max))
        ax.errorbar(range(n_channels), static_coefs, yerr=riro, fmt='o', elinewidth=4, capsize=6,
                    label='static-riro')
    # Add static component
    else:
        ax.scatter(range(n_channels), static_coefs, marker='o', label='static')

    # Draw a black line at y=0
    ax.hlines(0, 0, 1, transform=ax.get_yaxis_transform(), colors='k')

    delta_y = max_y - min_y
    # Add bounds on the graph
    if bounds is not None:
        len_vline_bounds = 0.01
        len_hline_bounds = 0.4
        if np.all(bounds[:, 0] == bounds[0, 0]) and np.all(bounds[:, 1] == bounds[0, 1]):
            ax.hlines(bounds[0, 0], 0 - len_hline_bounds, n_channels + len_hline_bounds, colors='r',
                      label='bounds', capstyle='projecting')
            ax.hlines(bounds[0, 1], 0 - len_hline_bounds, n_channels + len_hline_bounds, colors='r',
                      capstyle='projecting')
        else:
            # Channel 0 used for the legend
            # min
            ax.hlines(bounds[0, 0], -len_hline_bounds, len_hline_bounds, colors='r', label='bounds',
                      capstyle='projecting')
            ax.vlines(-len_hline_bounds, bounds[0, 0], bounds[0, 0] + (delta_y * len_vline_bounds), colors='r',
                      capstyle='projecting')
            ax.vlines(len_hline_bounds, bounds[0, 0], bounds[0, 0] + (delta_y * len_vline_bounds), colors='r',
                      capstyle='projecting')
            # max
            ax.hlines(bounds[0, 1], -len_hline_bounds, len_hline_bounds, colors='r', capstyle='projecting')
            ax.vlines(-len_hline_bounds, bounds[0, 1] - (delta_y * len_vline_bounds), bounds[0, 1], colors='r',
                      capstyle='projecting')
            ax.vlines(len_hline_bounds, bounds[0, 1] - (delta_y * len_vline_bounds), bounds[0, 1], colors='r',
                      capstyle='projecting')
            # All other channels
            for i_channel in range(1, n_channels):
                # min
                ax.hlines(bounds[i_channel, 0], i_channel - len_hline_bounds, i_channel + len_hline_bounds, colors='r',
                          capstyle='projecting')
                ax.vlines(i_channel - len_hline_bounds, bounds[i_channel, 0],
                          bounds[i_channel, 0] + (delta_y * len_vline_bounds), colors='r', capstyle='projecting')
                ax.vlines(i_channel + len_hline_bounds, bounds[i_channel, 0],
                          bounds[i_channel, 0] + (delta_y * len_vline_bounds), colors='r', capstyle='projecting')
                # max
                ax.hlines(bounds[i_channel, 1], i_channel - len_hline_bounds, i_channel + len_hline_bounds, colors='r',
                          capstyle='projecting')
                ax.vlines(i_channel - len_hline_bounds, bounds[i_channel, 1] - (delta_y * len_vline_bounds),
                          bounds[i_channel, 1], colors='r', capstyle='projecting')
                ax.vlines(i_channel + len_hline_bounds, bounds[i_channel, 1] - (delta_y * len_vline_bounds),
                          bounds[i_channel, 1], colors='r', capstyle='projecting')
    # Set the extent of the plot
    ax.set(ylim=(min_y - (0.05 * delta_y), max_y + (0.05 * delta_y)), xlim=(-0.75, n_channels - 0.25),
           xticks=range(n_channels))
    ax.legend()

    ax.set_title(f"Slices: {slice_number}, Total static current: {np.abs(static_coefs).sum()}")
    ax.set_xlabel('Channels')
    ax.set_ylabel(f"Coefficients {units}")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', nargs=1, type=click.Path(exists=True), required=True,
              help="4d volume where 4th dimension was acquired with different shim values")
@click.option('--mask', 'fname_mask', type=click.Path(exists=True), required=False,
              help="Mask defining the spatial region to shim. If no mask is provided, all voxels of the input will be "
                   "considered.")
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(os.path.abspath(os.curdir), 'shim_index.txt'),
              show_default=True, help="Filename to output shim text file.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def max_intensity(fname_input, fname_mask, fname_output, verbose):
    """ Find indexes of the 4th dimension of the input volume that has the highest signal intensity for each slice.
        Based on: https://onlinelibrary.wiley.com/doi/10.1002/hbm.26018

    """
    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    # Load the input file
    nii_input = nib.load(fname_input)

    # Load the mask
    if fname_mask is None:
        nii_mask = None
    else:
        nii_mask = nib.load(fname_mask)

    # Shim
    # Output with 1 index
    index_per_slice = shim_max_intensity(nii_input, nii_mask) + 1

    # Log the output (1 index)
    logger.info(f"Max intensity indexes: {index_per_slice}")

    # Write to a text file
    n_slices = len(index_per_slice)
    with open(fname_output, 'w', encoding='utf-8') as f:
        f.write(f"{n_slices}\n")
        for i_slice in range(n_slices - 1):
            f.write(f"{index_per_slice[i_slice]} ")
        f.write(f"{index_per_slice[n_slices - 1]}")

    logger.info(f"Txt file is located here:\n{fname_output}")


b0shim_cli.add_command(gradient_realtime)
b0shim_cli.add_command(dynamic)
b0shim_cli.add_command(realtime_dynamic)
b0shim_cli.add_command(max_intensity)
