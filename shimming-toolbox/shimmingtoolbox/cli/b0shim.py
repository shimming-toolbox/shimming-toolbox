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

from shimmingtoolbox import __config_scanner_constraints__, __config_custom_coil_constraints__
from shimmingtoolbox.cli.realtime_shim import gradient_realtime
from shimmingtoolbox.coils.coil import Coil, ScannerCoil, get_scanner_constraints, restrict_to_orders
from shimmingtoolbox.coils.spher_harm_basis import channels_per_order, reorder_shim_to_scaling_ge, SPH_HARMONICS_TITLES
from shimmingtoolbox.load_nifti import get_isocenter, is_fatsat_on
from shimmingtoolbox.pmu import PmuResp, PmuExt, PmuRespLog, PmuExtLog
from shimmingtoolbox.shim.sequencer import ShimSequencer, RealTimeSequencer
from shimmingtoolbox.shim.sequencer import shim_max_intensity, define_slices
from shimmingtoolbox.shim.sequencer import extend_fmap_to_kernel_size, parse_slices, new_bounds_from_currents
from shimmingtoolbox.utils import create_output_dir, set_all_loggers, timeit
from shimmingtoolbox.shim.shim_utils import phys_to_gradient_cs, shim_to_phys_cs
from shimmingtoolbox.shim.shim_utils import ScannerShimSettings

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AVAILABLE_ORDERS = [-1, 0, 1, 2, 3]


@click.group(context_settings=CONTEXT_SETTINGS,
             help="Shim according to the specified algorithm as an argument e.g. st_b0shim xxxxx")
def b0shim_cli():
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils', nargs=2, multiple=True, type=(click.Path(exists=True),
                                                               click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If using the scanner's gradient/shim coils, the coil profiles "
                   "must be in Hz/unit_shim and fieldmaps must be in Hz. If you want to shim using the scanner's "
                   "gradient/shim coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__config_custom_coil_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Static B0 fieldmap.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Anatomical image to apply the correction onto.")
@click.option('--mask', 'fname_mask_anat', type=click.Path(exists=True), required=False,
              help="Mask defining the spatial region to shim.")
@click.option('--scanner-coil-order', 'scanner_coil_order', type=click.STRING, default='-1',
              show_default=True,
              help="Spherical harmonics orders to be used in optimization. "
                   f"Available orders: {AVAILABLE_ORDERS}. "
                   "Orders should be writen with a coma separating the values. (i.e. 0,1,2)"
                   "The 0th order is the f0 frequency.")
@click.option('--scanner-coil-constraints', 'fname_sph_constr', type=click.Path(), default="",
              help=f"Constraints for the scanner coil. Example file located: {__config_scanner_constraints__}")
@click.option('--slices', type=click.Choice(['interleaved', 'ascending', 'descending', 'volume', 'auto']),
              required=False,
              default='auto', show_default=True,
              help="Define the slice ordering. If set to 'auto', automatically parse the target image.")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1,
              show_default=True,
              help="Number of slices per shimmed group. Used when '--slices' is not set to 'auto'. For example, if the "
                   "'--slice-factor' value is '3', then with the 'sequential' mode ('ascending' or 'descending'), "
                   "shimming will be performed independently on the following groups: {0,1,2}, {3,4,5}, etc. With the "
                   "mode 'interleaved', "
                   "it will be: {0,2,4}, {1,3,5}, etc.")
@click.option('--optimizer-method', 'method', required=False, default='quad_prog', show_default=True,
              type=click.Choice(['least_squares', 'pseudo_inverse', 'quad_prog', 'bfgs']),
              help="Method used by the optimizer. LS and QP will respect the constraints, "
              "BFGS method only accepts constraints for each channel (not constraints on the total current), "
              "PS will not respect any constraints")
@click.option('--regularization-factor', 'reg_factor', type=click.FLOAT, required=False, default=0.0,
              show_default=True,
              help="Regularization factor for the current when optimizing. A higher coefficient will penalize higher "
                   "current values while 0 provides no regularization. Not relevant for 'pseudo-inverse' "
                   "optimizer_method.")
@click.option('--optimizer-criteria', 'opt_criteria',
              type=click.Choice(['mse', 'mae', 'rmse', 'grad', 'ps_huber']), required=False,
              default='mse', show_default=True,
              help="Criteria of optimization for the optimizer 'least_squares' and 'bfgs'. "
                   "mse: Mean Squared Error, mae: Mean Absolute Error, ps_huber: pseudo huber cost function, "
                   "grad: Signal Loss, grad: mse of Bz + weighting X mse of Grad Z, relevant for signal recovery, "
                   "rmse: Root Mean Squared Error. Not relevant for 'pseudo_inverse' --optimizer-method.")
@click.option('--weighting-signal-loss', 'w_signal_loss', type=click.FLOAT, required=False, default=None,
              show_default=True,
              help="weighting for signal loss recovery. Since there is generally a compromise between B0 inhomogeneity"
                   " and gradient in z direction (i.e., signal loss recovery), a higher coefficient will put more "
                   "weights to recover the signal loss over the B0 inhomogeneity."
                   " This parameter can be used with the Least Squares optimization and the mse or rmse criteria.\n"
                   "The optimal value for mse is around 0.01\n"
                   "The optimal value for rmse is around 10")
@click.option('--weighting-signal-loss-xy', 'w_signal_loss_xy', type=click.FLOAT, required=False,
              default=None, show_default=True,
              help="weighting for signal loss recovery for the X and Y gradients. Since there is generally a "
                   "compromise between B0 inhomogeneity"
                   " and Gradient in z (through slice), x, y (phase and readout) direction (i.e., signal loss recovery)"
                   ", a higher coefficient will put more weights to recover the signal loss over the B0 inhomogeneity.")
@click.option('--mask-dilation-kernel-size', 'dilation_kernel_size', type=click.INT, required=False,
              default='3', show_default=True,
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
                                 'slicewise-hrd', 'chronological-hrd']),
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
                                      "'-hrd' to output a human readable file.")
@click.option('--output-value-format', 'output_value_format', type=click.Choice(['delta', 'absolute']), default='delta',
              show_default=True,
              help="Coefficient values for the scanner coil. delta: Outputs the change of shim coefficients. "
                   "absolute: Outputs the absolute coefficient by taking into account the current shim settings. "
                   "This is effectively initial + shim. Scanner coil coefficients will be in the Shim coordinate "
                   "system unless the option --output-file-format is set to gradient. The delta value format should be "
                   "used in that case.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
@timeit
def dynamic(fname_fmap, fname_anat, fname_mask_anat, method, opt_criteria, slices, slice_factor, coils,
            dilation_kernel_size, scanner_coil_order, fname_sph_constr, fatsat, path_output, o_format_coil,
            o_format_sph, output_value_format, reg_factor, w_signal_loss, w_signal_loss_xy, verbose):
    """ Static shim by fitting a fieldmap. Use the option --optimizer-method to change the shimming algorithm used to
    optimize. Use the options --slices and --slice-factor to change the shimming order/size of the slices.

    Example of use: st_b0shim dynamic --coil coil1.nii coil1_constraints.json --coil coil2.nii coil2_constraints.json
    --fmap fmap.nii --anat anat.nii --mask mask.nii --optimizer-method least_squares
    """

    logger.info(f"Output value format: {output_value_format}, o_format_coil: {o_format_coil}")

    # Set logger level
    set_all_loggers(verbose)

    # Parse scanner_coil_order
    scanner_coil_order = parse_orders(scanner_coil_order)

    # Prepare the output
    create_output_dir(path_output)

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

    # Get the EPI echo time and set signal recovery optimizer criteria if w signal loss is set
    if (w_signal_loss is not None) or (w_signal_loss_xy is not None):
        if opt_criteria not in ['mse', 'rmse']:
            raise ValueError("Signal loss weighting is only available with the mse optimization criteria")

        opt_criteria += '_signal_recovery'
        epi_te = json_anat_data.get('EchoTime')

        if w_signal_loss is None:
            w_signal_loss = 0
        if w_signal_loss_xy is None:
            w_signal_loss_xy = 0
    else:
        epi_te = None

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
    fname_json = fname_fmap.rsplit('.nii', 1)[0] + '.json'
    # Read from json file
    if os.path.isfile(fname_json):
        with open(fname_json) as json_file:
            json_fm_data = json.load(json_file)
    else:
        raise OSError("Missing fieldmap json file")

    # Error out for unsupported inputs. If file format is in gradient CS, it must be 1st order and the output format be
    # delta. Only Siemens gradient coordinate system has been defined
    if 'hrd' in o_format_sph:
        if output_value_format != 'delta':
            raise ValueError(f"Unsupported output value format: {output_value_format} for output file format: "
                             f"{o_format_sph}")
        if not set(scanner_coil_order).issubset({0, 1}):
            raise ValueError(f"Unsupported scanner coil order: {scanner_coil_order} for output file format: "
                             f"{o_format_sph}. Supported orders are: [0, 1], [1], [0]")
        if json_fm_data.get('Manufacturer') != 'Siemens':
            raise NotImplementedError(f"Unsupported manufacturer: {json_fm_data.get('Manufacturer')} for output file"
                                      f"format: {o_format_sph}")

    # Find the isocenter
    isocenter_fm = get_isocenter(json_fm_data)
    isocenter_anat = get_isocenter(json_anat_data)
    if not np.all(np.isclose(isocenter_fm, isocenter_anat)):
        raise ValueError("Table position in the field map and target image are not the same.")

    # Read the current shim settings from the scanner
    scanner_shim_settings = ScannerShimSettings(json_fm_data, orders=scanner_coil_order)
    options = {'scanner_shim': scanner_shim_settings.shim_settings}

    # Load the coils
    list_coils = load_coils(coils, scanner_coil_order, fname_sph_constr, nii_fmap, options['scanner_shim'],
                             json_fm_data)

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    if slices == 'auto':
        list_slices = parse_slices(fname_anat)
    else:
        list_slices = define_slices(n_slices, slice_factor, slices, json_fm_data.get('SoftwareVersions'))
    logger.info(f"The slices to shim are:\n{list_slices}")
    # Get shimming coefficients
    # 1 ) Create the Shimming sequencer object
    sequencer = ShimSequencer(nii_fmap_orig, json_fm_data,
                              nii_anat, json_anat_data,
                              nii_mask_anat,
                              list_slices, list_coils,
                              method=method,
                              opt_criteria=opt_criteria,
                              mask_dilation_kernel='sphere',
                              mask_dilation_kernel_size=dilation_kernel_size,
                              reg_factor=reg_factor,
                              w_signal_loss=w_signal_loss,
                              w_signal_loss_xy=w_signal_loss_xy,
                              epi_te=epi_te,
                              path_output=path_output)

    # 2) Launch shim sequencer
    coefs = sequencer.shim()
    # Output
    # Load output options
    options['fatsat'] = get_fatsat_option(json_anat_data, fatsat)

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

            # If outputting in the gradient CS, it must be specific orders, it must be in the delta CS and Siemens
            # The check has already been done earlier in the program to avoid processing and throw an error afterwards.
            # Therefore, we can only check for the o_format_sph.
            if 'hrd' in o_format_sph:
                logger.debug("Converting Siemens scanner coil from Shim CS (LAI) to Gradient CS")

                coefs_coil = coefs_to_dict(coefs_coil, scanner_coil_order, manufacturer)

            else:
                # If the output format is absolute, add the initial coefs
                if output_value_format == 'absolute':
                    initial_coefs = scanner_shim_settings.concatenate_shim_settings(scanner_coil_order)
                    for i_channel in range(n_channels):
                        # abs_coef = delta + initial
                        coefs_coil[:, i_channel] = coefs_coil[:, i_channel] + initial_coefs[i_channel]

                    list_fname_output += _save_to_text_file(coil, coefs_coil, list_slices, path_output,
                                                                   o_format_sph, options, coil_number=i_coil,
                                                                   default_coefs=initial_coefs)
                    continue

            list_fname_output += _save_to_text_file(coil, coefs_coil, list_slices, path_output, o_format_sph,
                                                           options, coil_number=i_coil)

        else:
            list_fname_output += _save_to_text_file(coil, coefs_coil, list_slices, path_output, o_format_coil,
                                                           options, coil_number=i_coil)

    logger.info(f"Coil txt file(s) are here:\n{os.linesep.join(list_fname_output)}")
    logger.info(f"Plotting figure(s)")
    sequencer.eval(coefs)
    logger.info(f" Plotting currents")

    if logger.level <= getattr(logging, 'DEBUG'):
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


def _save_to_text_file(coil, coefs, list_slices, path_output, o_format, options, coil_number,
                              default_coefs=None, mean_pressure=None):
    """o_format can either be 'slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil', 'gradient'"""

    logger.info(f"Saving to text file with format: {o_format}")
    n_channels = coil.dim[3]
    list_fname_output = []
    if o_format[-5:] == '-coil':

        fname_output = os.path.join(path_output, f"coefs_coil{coil_number}_{coil.name}.txt")
        if options['fatsat']:
            fname_output_no_fatsat = os.path.join(path_output, f"coefs_coil{coil_number}_{coil.name}_no_fatsat.txt")
            f_no_fatsat = open(fname_output_no_fatsat, 'w', encoding='utf-8')

        # Print the average shim coefficients without considering slices with 0s
        logger.info(f"Average shim coefficients for coil {coil.name} without considering slices with 0s: "
                    f"{np.mean(np.sum(abs(coefs), axis=1, where=(coefs != 0)), axis=0)}")

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
                        if options['fatsat']:
                            f_no_fatsat.write(f"{coefs[i_shim, i_channel]:.6f}, ")

                    f.write("\n")
                    if options['fatsat']:
                        f_no_fatsat.write("\n")

            elif o_format == 'slicewise-coil':
                # Output per slice, output all channels for a particular slice, then repeat
                # Assumes all slices are in list_slices once which is the case for ascending, descending, interleaved and
                # volume
                n_slices = np.sum([len(a_shim) for a_shim in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(a_shim) for a_shim in list_slices if i_slice in a_shim][0]
                    for i_channel in range(n_channels):
                        f.write(f"{coefs[i_shim, i_channel]:.6f}, ")
                    f.write("\n")

        if options['fatsat']:
            f_no_fatsat.close()
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

    else:  # o_format == 'human readable':
        if mean_pressure is None:
            fname_output = os.path.join(path_output, f"scanner_shim.txt")
        else:
            fname_output = os.path.join(path_output, f"scanner_shim_riro.txt")

        # Create column names: "orderX_channelY"
        column_names = ['f0', 'Gx', 'Gy', 'Gz']
        orders = [0, 1]

        # Transform dict into usable array (nb_slices, n_channels)
        arrays = [coefs.get(order, np.zeros((len(list_slices), 2*order+1))) / 1000 \
            if order == 1 else coefs.get(order, np.zeros((len(list_slices), 2*order+1))) for order in orders]
        coefs_array = np.hstack(arrays)

        if "slicewise" in o_format:
            # reorder according to list_slices
            # Build a reverse mapping: from list_slices to target positions
            inverse_slice_order = np.argsort([tup[0] for tup in list_slices])
            # Reorder the array
            coefs_array = coefs_array[inverse_slice_order, :]

        # Compute column widths
        # 1. Get max formatted value length in each column
        formatted_values = []
        col_widths = []

        for col_idx in range(coefs_array.shape[1]):
            col_vals = coefs_array[:, col_idx]
            formatted_col = [f"{val:.6f}" for val in col_vals]
            formatted_values.append(formatted_col)
            max_val_len = max(len(s) for s in formatted_col)
            col_name_len = len(column_names[col_idx])
            col_width = max(max_val_len, col_name_len)
            col_widths.append(col_width)

        # Write to file manually
        with open(fname_output, 'w') as f:
            if mean_pressure is not None:
                f.write(f"Mean pressure = {mean_pressure:.2f}\n")
            # Write header (centered titles)
            header_cells = [column_names[i].center(col_widths[i]) for i in range(len(column_names))]
            header = ' | '.join(header_cells)
            f.write(header + '\n')

            # Write each row of shim values (right-aligned)
            nb_rows = coefs_array.shape[0]
            for row_idx in range(nb_rows):
                row_cells = [
                    formatted_values[col_idx][row_idx].rjust(col_widths[col_idx])
                    for col_idx in range(len(column_names))
                ]
                row_str = ' | '.join(row_cells)
                f.write(row_str + '\n')

        list_fname_output.append(os.path.abspath(fname_output))

    return list_fname_output


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--coil', 'coils_static', nargs=2, multiple=True,
              type=(click.Path(exists=True), click.Path(exists=True)),
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__config_custom_coil_constraints__}")
@click.option('--coil-riro', 'coils_riro', nargs=2, multiple=True,
              type=(click.Path(exists=True), click.Path(exists=True)), required=False,
              help="Pair of filenames containing the coil profiles followed by the filename to the constraints "
                   "e.g. --coil a.nii cons.json. If you have more than one coil, use this option more than once. "
                   "The coil profiles and the fieldmaps (--fmap) must have matching units (if fmap is in Hz, the coil "
                   "profiles must be in Hz/unit_shim). If this option is used, these coil profiles will be used for "
                   "the RIRO optimization, otherwise, the coils from the --coil options will be used."
                   "If you only want to shim using the scanner's gradient/shim "
                   "coils, use the `--scanner-coil-order` option. For an example of a constraint file, "
                   f"see: {__config_custom_coil_constraints__}")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Timeseries of B0 fieldmap.")
@click.option('--anat', 'fname_anat', type=click.Path(exists=True), required=True,
              help="Anatomical image to apply the correction onto.")
@click.option('--resp', 'fname_resp', type=click.Path(exists=True), required=True,
              help="Siemens respiratory file containing pressure data. Supported extensions: '.resp', '.log'.")
@click.option('--trigs', 'fname_ext', type=click.Path(exists=True), required=False,
              help="Siemens external trigger file containing pressure data. Supported extensions: '.ext', '.log'.")
@click.option('--time-offset', 'time_offset', type=click.STRING, required=False, default='0',
              help="Time offset (ms) between the respiratory recording and the acquired time in the DICOMs.")
@click.option('--mask-static', 'fname_mask_anat_static', type=click.Path(exists=True), required=False,
              help="Mask defining the static spatial region to shim.")
@click.option('--mask-riro', 'fname_mask_anat_riro', type=click.Path(exists=True), required=False,
              help="Mask defining the time varying (i.e. RIRO, Respiration-Induced Resonance Offset) "
                   "region to shim.")
@click.option('--scanner-coil-order', 'scanner_coil_order_static', type=click.STRING, default='-1',
              show_default=True,
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
              help=f"Constraints for the scanner coil. Example file located: {__config_scanner_constraints__}")
@click.option('--slices', type=click.Choice(['interleaved', 'ascending', 'descdending', 'volume', 'auto']),
              required=False,
              default='auto', show_default=True,
              help="Define the slice ordering. If set to 'auto', automatically parse the target image.")
@click.option('--slice-factor', 'slice_factor', type=click.INT, required=False, default=1, show_default=True,
              help="Number of slices per shimmed group. Used when '--slices' is not set to 'auto'. For example, if the "
                   "'--slice-factor' value is '3', then with the 'sequential' ('ascending' or 'descending') mode, "
                   "shimming will be performed independently on the following groups: {0,1,2}, {3,4,5}, etc. With the "
                   "mode 'interleaved', "
                   "it will be: {0,2,4}, {1,3,5}, etc.")
@click.option('--optimizer-method', 'method', required=False, default='quad_prog', show_default=True,
              type=click.Choice(['least_squares', 'pseudo_inverse', 'quad_prog', 'bfgs']),
              help="Method used by the optimizer. LS and QP will respect the constraints, "
              "BFGS method only accepts constraints for each channel (not constraints on the total current), "
              "PS will not respect any constraints")
@click.option('--optimizer-criteria', 'opt_criteria', type=click.Choice(['mse', 'mae', 'grad', 'rmse']), required=False,
              default='mse', show_default=True,
              help="Criteria of optimization for the optimizer 'least_squares' and 'bfgs'. "
                   "mse: Mean Squared Error, mae: Mean Absolute Error, ps_huber: pseudo huber cost function, "
                   "rmse: Root Mean Squared Error. Not relevant for 'pseudo_inverse' or 'quad_prog' "
                   "--optimizer-method.")
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
              type=click.Choice(['slicewise-ch', 'chronological-ch', 'chronological-coil', 'slicewise-coil']),
              default='slicewise-ch', show_default=True,
              help="Syntax used to describe the sequence of shim events. "
                   "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                   "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                   "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                   "captured by the controller of the shim amplifier. For 'XXXXX-ch', "
                   "there will be one output file per coil channel (coil1_ch1.txt, coil1_ch2.txt, etc.). The static, "
                   "time-varying and mean pressure are encoded in the columns of each file. For XXXXX-coil, there will "
                   "be a single file per coil. The static and time-varying coefficients are encoded one after the "
                   "other as columns (static-ch1, rt-ch1, static-ch2, rt-ch2, etc.). The mean pressure is encoded as "
                   "the last row.")
@click.option('--output-file-format-scanner', 'o_format_sph',
              type=click.Choice(['slicewise-ch', 'slicewise-coil', 'chronological-ch', 'chronological-coil',
                                 'slicewise-hrd', 'chronological-hrd']),
              default='slicewise-ch',
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
                                      "'-hrd' to output a human readable file.")
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
                     o_format_coil, o_format_sph, output_value_format, reg_factor, time_offset, fname_ext, verbose):
    """ Realtime shim by fitting a fieldmap to a pressure monitoring unit. Use the option --optimizer-method to change
    the shimming algorithm used to optimize. Use the options --slices and --slice-factor to change the shimming
    order/size of the slices.

    Example of use: st_b0shim realtime-dynamic --coil coil1.nii coil1_constraints.json --coil coil2.nii coil2_constraints.json
    --fmap fmap.nii --anat anat.nii --mask-static mask.nii --resp trace.resp --optimizer-method least_squares
    """

    logger.info(f"Output value format: {output_value_format}, o_format_coil: {o_format_coil}")

    # Set logger level
    set_all_loggers(verbose)

    # Set coils and scanner order for riro if none were indicated
    if scanner_coil_order_riro is None:
        scanner_coil_order_riro = scanner_coil_order_static

    scanner_coil_order_static = parse_orders(scanner_coil_order_static)
    scanner_coil_order_riro = parse_orders(scanner_coil_order_riro)

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
        nii_mask_anat_riro = copy.deepcopy(nii_mask_anat_static)

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
    if 'hrd' in o_format_sph:
        if output_value_format == 'absolute':
            raise ValueError(f"Unsupported output value format: {output_value_format} for output file format: "
                             f"{o_format_sph}")
        if not set(scanner_coil_order_static).issubset({0, 1}):
            raise ValueError(f"Unsupported scanner coil order: {scanner_coil_order_static} for output file format: "
                             f"{o_format_sph}")
        if not set(scanner_coil_order_riro).issubset({0, 1}):
            raise ValueError(f"Unsupported scanner coil order: {scanner_coil_order_riro} for output file format: "
                             f"{o_format_sph}")
        if json_fm_data['Manufacturer'] != 'Siemens':
            raise ValueError(f"Unsupported manufacturer: {json_fm_data['manufacturer']} for output file format: "
                             f"{o_format_sph}")

    # Find the isocenter
    isocenter_fm = get_isocenter(json_fm_data)
    isocenter_anat = get_isocenter(json_anat_data)
    if isocenter_fm is None or isocenter_anat is None or not np.all(np.isclose(isocenter_fm, isocenter_anat)):
        raise ValueError("Table position in the field map and target image are not the same.")

    # Read the current shim settings from the scanner
    all_scanner_orders = set(scanner_coil_order_static).union(set(scanner_coil_order_riro))
    scanner_shim_settings = ScannerShimSettings(json_fm_data, orders=all_scanner_orders)
    options = {'scanner_shim': scanner_shim_settings.shim_settings}

    # Load the coils
    list_coils_static = load_coils(coils_static, scanner_coil_order_static, fname_sph_constr, nii_fmap,
                                    options['scanner_shim'], json_fm_data)
    list_coils_riro = load_coils(coils_riro, scanner_coil_order_riro, fname_sph_constr, nii_fmap,
                                  options['scanner_shim'], json_fm_data)

    if logger.level <= getattr(logging, 'DEBUG'):
        # Save inputs
        list_fname = [fname_fmap, fname_anat, fname_mask_anat_static, fname_mask_anat_riro]
        _save_nii_to_new_dir(list_fname, path_output)

    # Get the shim slice ordering
    n_slices = nii_anat.shape[2]
    if slices == 'auto':
        list_slices = parse_slices(fname_anat)
    else:
        list_slices = define_slices(n_slices, slice_factor, slices, json_fm_data.get('SoftwareVersions'))

    logger.info(f"The slices to shim are: {list_slices}")

    # Load PMU
    if time_offset == 'auto':
        is_pmu_time_offset_auto = True
        time_offset = 0
    else:
        is_pmu_time_offset_auto = False
        time_offset = round(int(time_offset))

    if os.path.splitext(fname_resp)[1] == '.log':
        pmu = PmuRespLog(fname_resp, time_offset=time_offset)
    else:
        pmu = PmuResp(fname_resp, time_offset=time_offset)
    if fname_ext is not None:
        # Load external trigger file if provided
        if os.path.splitext(fname_resp)[1] == '.log':
            pmu_ext = PmuExtLog(fname_ext)
        else:
            pmu_ext = PmuExt(fname_ext)
    else:
        pmu_ext = None

    # 1 ) Create the real time pmu sequencer object
    sequencer = RealTimeSequencer(nii_fmap_orig, json_fm_data, nii_anat, nii_mask_anat_static,
                                  nii_mask_anat_riro,
                                  list_slices, pmu, list_coils_static, list_coils_riro,
                                  method=method,
                                  opt_criteria=opt_criteria,
                                  mask_dilation_kernel='sphere',
                                  mask_dilation_kernel_size=dilation_kernel_size,
                                  reg_factor=reg_factor,
                                  path_output=path_output,
                                  pmu_ext=pmu_ext,
                                  is_pmu_time_offset_auto=is_pmu_time_offset_auto)
    # 2) Launch the sequencer
    coefs_static, coefs_riro, mean_p, p_rms = sequencer.shim()

    # Output
    # Load output options
    options['fatsat'] = get_fatsat_option(json_anat_data, fatsat)

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
            if 'hrd' in o_format_sph:
                logger.debug("Converting Siemens scanner coil from Shim CS (LAI) to Gradient CS")

                coefs_coil_static = coefs_to_dict(coefs_static, scanner_coil_order_static,
                                                   json_anat_data['Manufacturer'])
                coefs_coil_riro = coefs_to_dict(coefs_riro, scanner_coil_order_riro,
                                                 json_anat_data['Manufacturer'])
                list_fname_output += _save_to_text_file(coil, coefs_coil_static, list_slices, path_output,
                                                        o_format_sph, options, coil_number=i_coil)
                list_fname_output += _save_to_text_file(coil, coefs_coil_riro, list_slices, path_output,
                                                        o_format_sph, options, coil_number=i_coil,
                                                        mean_pressure=mean_p)
            else:
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

        elif o_format == 'chronological-coil':
            fname_output = os.path.join(path_output, f"coefs_coil{coil_number}_{coil.name}.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                for i_shim in range(len(list_slices)):
                    if options['fatsat']:
                        for i_channel in range(n_channels):
                            if default_st_coefs is None:
                                # Output 0 (delta)
                                f.write(f"{0:.1f}, {0:.1f}, ")
                            else:
                                # Output initial coefs (absolute)
                                f.write(f"{default_st_coefs[i_channel]:.1f}, {0:.1f}, ")
                        f.write("\n")

                    for i_channel in range(n_channels):
                        if currents_static is not None:
                            f.write(f"{currents_static[i_shim, i_channel]:.6f}, ")
                        if currents_riro is not None:
                            f.write(f"{currents_riro[i_shim, i_channel]:.12f}, ")
                    f.write("\n")
                f.write(f"{mean_p:.4f},\n")

        elif o_format == 'slicewise-coil':
            fname_output = os.path.join(path_output, f"coefs_coil{coil_number}_{coil.name}.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                # Each row will have one coef representing the static, riro and mean_p in slicewise order
                n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]
                    for i_channel in range(n_channels):
                        if currents_static is not None:
                            f.write(f"{currents_static[i_shim, i_channel]:.6f}, ")
                        if currents_riro is not None:
                            f.write(f"{currents_riro[i_shim, i_channel]:.12f}, ")
                    f.write("\n")
                f.write(f"{mean_p:.4f},\n")

        else:  # o_format == 'gradient':
            f0 = 'f0'
            gradients = ['x', 'y', 'z']
            if n_channels == 1:
                name = {0: f0}
            elif n_channels == 3:
                name = {0: gradients[0],
                        1: gradients[1],
                        2: gradients[2]}
            elif n_channels == 4:
                name = {0: f0,
                        1: gradients[0],
                        2: gradients[1],
                        3: gradients[2]}
            else:
                raise RuntimeError("Gradient output format should only be used with 1st order scanner coils")

            fname_output = os.path.join(path_output, f"{name[i_channel]}shim_gradients.txt")
            with open(fname_output, 'w', encoding='utf-8') as f:
                n_slices = np.sum([len(a_tuple) for a_tuple in list_slices])
                for i_slice in range(n_slices):
                    i_shim = [list_slices.index(i) for i in list_slices if i_slice in i][0]

                    if name[i_channel] == f0:
                        # f0, Output is in Hz
                        if currents_static is not None:
                            f.write(f"corr_vec[0][{i_slice}]= "
                                    f"{currents_static[i_shim, i_channel]:.6f}\n")
                        if currents_riro is not None:
                            f.write(f"corr_vec[1][{i_slice}]= "
                                    f"{currents_riro[i_shim, i_channel]:.12f}\n")
                        f.write(f"corr_vec[2][{i_slice}]= {mean_p:.3f}\n")

                    elif name[i_channel] in gradients:
                        # For Gx, Gy, Gz: Divide by 1000 for mT/m
                        if currents_static is not None:
                            f.write(f"corr_vec[0][{i_slice}]= "
                                    f"{currents_static[i_shim, i_channel] / 1000:.6f}\n")
                        if currents_riro is not None:
                            f.write(f"corr_vec[1][{i_slice}]= "
                                    f"{currents_riro[i_shim, i_channel] / 1000:.12f}\n")
                        f.write(f"corr_vec[2][{i_slice}]= {mean_p:.3f}\n")
                    else:
                        raise RuntimeError("Unsupported name")

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


def load_coils(coils, orders, fname_constraints, nii_fmap, scanner_shim_settings, json_fm_data):
    """ Loads the Coil objects from filenames

    Args:
        coils (list): List of tuples(fname_nii, fname_json) of coil profiles and constraints
        orders (list): Orders of the scanner coils (0 or 1 or 2)
        fname_constraints (str): Filename of the constraints of the scanner coils
        nii_fmap (nib.Nifti1Image): Nibabel object of the fieldmap
        scanner_shim_settings (dict): Dictionary containing the shim settings of the scanner ('0', '1', '2')
        json_fm_data (dict): BIDS JSON sidecar as a dictionary

    Returns:
        list: List of Coil objects containing the custom coils followed by the scanner coil if requested
    """

    manufacturer = json_fm_data.get('Manufacturer')
    manufacturers_model_name = json_fm_data.get('ManufacturersModelName')
    if manufacturers_model_name is not None:
        manufacturers_model_name = manufacturers_model_name.replace(' ', '_')

    list_coils = []

    # Load custom coils
    # Load custom coils
    for coil in coils:
        nii_coil_profiles = nib.load(coil[0])
        coil_data = nii_coil_profiles.get_fdata()

        # If 3D, extend to 4D by adding singleton dimension
        if coil_data.ndim == 3:
            coil_data = coil_data[..., np.newaxis]

        with open(coil[1]) as json_file:
            constraints = json.load(json_file)
        list_coils.append(Coil(coil_data, nii_coil_profiles.affine, constraints))

    if len(list_coils) != len(set(list_coils)):
        raise ValueError("Coils must be unique. Make sure different coils have different names.")

    # Create the spherical harmonic coil profiles of the scanner
    if -1 not in orders:
        # Todo: Skip loading constraints if the algo does not support constraints?
        if os.path.isfile(fname_constraints):
            with open(fname_constraints) as json_file:
                external_contraints = json.load(json_file)
            scanner_contraints = get_scanner_constraints(manufacturers_model_name, orders, manufacturer,
                                                         scanner_shim_settings, external_contraints)
        else:
            scanner_contraints = get_scanner_constraints(manufacturers_model_name, orders, manufacturer,
                                                         scanner_shim_settings)

        isocenter = get_isocenter(json_fm_data)
        scanner_coil = ScannerCoil(nii_fmap.shape[:3], nii_fmap.affine, scanner_contraints, orders,
                                   manufacturer=manufacturer, isocenter=isocenter)
        list_coils.append(scanner_coil)

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


def get_fatsat_option(json_anat, fatsat):
    """ Return if the fat saturation option should be turned on or off.
        This function mainly exists to resolve the 'auto' case

    Args:
        json_anat (dict): BIDS Json sidecar
        fatsat (str): String containing either : 'yes', 'no' or 'auto'

    Returns:
        bool: Whether to activate fatsat or not
    """
    fatsat_option = False

    if fatsat == 'auto':
        fatsat_option = is_fatsat_on(json_anat)
    elif fatsat == 'yes':
        fatsat_option = True
    elif fatsat == 'no':
        pass
    else:
        raise ValueError(f"Invalid fatsat option: {fatsat}. Must be 'yes', 'no' or 'auto'.")

    return fatsat_option


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--slices', required=True,
              help="Enter the total number of slices. Also accepts a path to an anatomical file to determine the "
                   "number of slices automatically. (Looks at 3rd dim)")
@click.option('--factor', required=True, type=click.INT,
              help="Number of slices per shim")
@click.option('--method', type=click.Choice(['interleaved', 'ascending', 'descending', 'volume']), required=True,
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


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', nargs=1, type=click.Path(exists=True), required=True,
              help="Text file containing the shim coefficients. Supported formats: .txt")
@click.option('-i2', '--input2', 'fname_input2', nargs=1, type=click.Path(exists=True), required=True,
              help="Text file containing the shim coefficients. Supported formats: .txt")
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(os.path.abspath(os.curdir), 'shim_coefs.txt'),
              show_default=True, help="Filename to output shim text file.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
def add_shim_coefs(fname_input, fname_input2, fname_output, verbose):
    """ Combine the shim coefficients from two files into a single file."""
    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    coefs1 = read_txt_file(fname_input)
    coefs2 = read_txt_file(fname_input2)

    if coefs1.shape == coefs2.shape:
        coefs = coefs1 + coefs2
    else:
        raise ValueError("The number of shim events and/or the number of channels is not the same in both text files")

    logger.debug(coefs1)
    logger.debug(coefs2)
    logger.debug(coefs)
    write_coefs_to_text_file(coefs, fname_output, 'slicewise')


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'fname_input', nargs=1, type=click.Path(exists=True), required=True,
              help="Text file containing the shim coefficients. Supported formats: .txt")
@click.option('--input-file-format', 'i_format',
              type=click.Choice(['volume', 'slicewise', 'chronological']), required=True,
              help="Syntax used to describe the sequence of shim events for a coil or coil channel. "
                   "Use 'slicewise' if the inputs in row 1, 2, 3, etc. are the shim coefficients for slice "
                   "1, 2, 3, etc. Use 'chronological' if the inputs in row 1, 2, 3, etc. are the shim value "
                   "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                   "captured by the controller of the shim amplifier. Use volume if the intput is a single set of shim "
                   "coefficients.")
@click.option('--output-file-format', 'o_format',
              type=click.Choice(['volume', 'slicewise', 'chronological', 'custom-cl']), required=True,
              help="Syntax used to describe the sequence of shim events for a coil or coil channel. "
                   "Use 'slicewise' to output in row 1, 2, 3, etc. the shim coefficients for slice "
                   "1, 2, 3, etc. Use 'chronological' to output in row 1, 2, 3, etc. the shim value "
                   "for trigger 1, 2, 3, etc. The trigger is an event sent by the scanner and "
                   "captured by the controller of the shim amplifier. 'custom-cl' is a custom format for a "
                   "collaborator. Use volume to output a single set of shim coefficients.")
@click.option('--target', 'fname_target', nargs=1, type=click.Path(exists=True), required=False,
              help="Target image the text file is based on. This is used to infer slice timing information when "
                   "converting between 'slicewise' and 'chronological'. It is also used to infer the number of slices "
                   "when converting from volume to any other format. Supported formats: .nii, .nii.gz")
@click.option('--reverse-slice-order', 'rev_slice_order', is_flag=True, default=False,
              help="Reverse the order of the slices. Only relevant for 'custom-cl'", required=False)
@click.option('--add-channels', 'to_add_channels',
              help="Add channels to the text file that are 0s. ", type=click.STRING, default='', required=False)
@click.option('-o', '--output', 'fname_output', type=click.Path(),
              default=os.path.join(os.path.abspath(os.curdir), 'shim_coefs.txt'),
              show_default=True, help="Filename to output shim text file.")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
def convert_shim_coefs_format(fname_input, i_format, o_format, fname_target, rev_slice_order, fname_output,
                              to_add_channels, verbose):
    """ Convert the shim coefficients from one format to another."""

    # Set logger level
    set_all_loggers(verbose)

    # Prepare the output
    create_output_dir(fname_output, is_file=True)

    if o_format == 'custom_cl' and i_format != 'slicewise':
        raise ValueError("Custom-cl output format is only compatible with slicewise input format")
    if i_format in ['chronological', 'volume'] or o_format == 'chronological':
        if fname_target is None:
            raise ValueError("The target image is required for the specified input/output formats")
        nii_target = nib.load(fname_target)

    coefs = read_txt_file(fname_input)

    to_add_channels = parse_add_channels(to_add_channels, 9)
    coefs = add_channels(coefs, to_add_channels)

    # convert coefs
    if i_format == 'volume':
        # convert to slice_wise
        coefs = np.repeat(coefs, nii_target.shape[2], axis=0)
    elif i_format == 'chronological':
        # convert to slice_wise
        slices = parse_slices(fname_target)
        tmp = np.zeros((nii_target.shape[2], coefs.shape[1]))
        for i_slice, slice in enumerate(slices):
            tmp[slice] = np.repeat(coefs[i_slice], len(slice))
        coefs = tmp

    # All coefficients should be in a slicewise format at this point
    # Convert from slicewise to the desired format

    if o_format == 'volume':
        # convert to volume
        for i_slice in range(coefs.shape[0]):
            if not np.all(coefs[i_slice] == coefs[0]):
                raise ValueError("All slices must have the same shim coefficients to convert to volume format")
        coefs = coefs[0]

    elif o_format == 'chronological':
        # convert to chronological
        slices = parse_slices(fname_target)
        tmp = np.zeros((len(slices), coefs.shape[1]))
        for i_shim in range(len(slices)):
            for i_slice in range(len(slices[i_shim])):
                if np.all(coefs[slices[i_shim][i_slice]] == coefs[slices[i_shim][0]]):
                    tmp[i_shim] = coefs[slices[i_shim][0]]
        coefs = tmp
    elif o_format == 'custom_cl':
        # Make sure there are 9 channels
        if coefs.shape[1] != 9:
            raise ValueError("The number of channels in one of the text files must be 9 for the custom-cl format")

        # Make sure the 2nd order shims are the same for all slices
        for i_shim in range(coefs.shape[0]):
            if not np.all(np.isclose(coefs[i_shim][4:], coefs[0][4:])):
                raise ValueError("The 2nd order shims must be the same for all slices to convert to 'custom-cl' format")

        # Send to write_coefs_to_text_file in a slice-wise format, the formatting is handled in that function

    write_coefs_to_text_file(coefs, fname_output, o_format, rev_slice_order)


def parse_add_channels(channels: str, n_channels: int):
    """
    Parse the channels to add to the shim coefficients
    Args:
        channels (str): String containing the channels to add
        n_channels (int): Number of channels that there currently is

    Returns:
        list: List of channels to add
    """
    channels = channels.split(',')
    try:
        if channels == ['']:
            return []
        channels = [int(channel) for channel in channels]
        channels.sort()
        if len(channels) + n_channels <= max(channels):
            raise ValueError(f"The provided channels to add would leave gaps in the channels")
        return channels
    except ValueError:
        raise ValueError(f"Invalid channels: {channels}\n Channels must be integers ")


def add_channels(coefs: np.array, channels: list):
    """
    Add channels to the shim coefficients
    Args:
        coefs (np.array): Shim coefficients (n_shims x n_channels)
        channels (list): List of channels to add

    Returns:
        np.array: Shim coefficients with added channels

    """
    if len(channels) == 0:
        return coefs
    for channel in channels:
        coefs = np.insert(coefs, channel, 0, axis=1)
    return coefs


def read_txt_file(fname_input):
    """
    Read the text file containing the shim coefficients
    Args:
        fname_input (str): Filename of the text file

    Returns:
        np.array: Array containing the shim coefficients
    """
    coefs = []
    with open(fname_input, 'r') as f:
        for i_line, line in enumerate(f):
            list_line = line.strip('\n').split(',')
            temp = []
            for i, value in enumerate(list_line):
                if value.strip(' ') != '':
                    temp.append(float(value.strip()))
            coefs.append(temp)
    n_lines = i_line + 1
    coefs = np.array(coefs)
    logger.debug(f"Reading text file. Number of shim events: {n_lines}, number of channels: {coefs.shape[1]}")
    return coefs


def write_coefs_to_text_file(coefs, fname_output, o_format, rev_slice_order=False):
    if o_format == 'slicewise' or o_format == 'chronological':
        with open(fname_output, 'w', encoding='utf-8') as f:
            for i_shim in range(coefs.shape[0]):
                for i_coef, coef in enumerate(coefs[i_shim]):
                    f.write(f"{coef:.6f},")
                    if i_coef != coefs.shape[1] - 1:
                        f.write(" ")
                f.write("\n")
    elif o_format == 'volume':
        with open(fname_output, 'w', encoding='utf-8') as f:
            for i_coef, coef in enumerate(coefs):
                f.write(f"{coef:.6f},")
                if i_coef != len(coefs) - 1:
                    f.write(" ")
    elif o_format == 'custom-cl':
        coefs[:, 0] *= -1
        if coefs.shape[1] != 9:
            raise ValueError("The number of channels in the text file must be 9 for the custom-cl format")
        with open(fname_output, 'w', encoding='utf-8') as f:
            f.write("(mA)%6s%11s%11s%11s%11s\n" % ("xy", "zy", "zx", "x2-y2", "z2"))
            ref = coefs[0][4:]

            for i_shim in range(coefs.shape[0]):
                if not np.all(np.isclose(coefs[i_shim][4:], ref)):
                    raise ValueError("The 2nd order shims must be the same for all slices")

            coefs[..., 4:] = reorder_shim_to_scaling_ge(coefs[..., 4:])

            f.write("%10s%11s%11s%11s%11s\n" % tuple(str(int(coefs[0][i_channel])) for i_channel in range(4, 9)))
            f.write("\n(G/cm)%6s%13s%13s%13s\n" % ("x", "y", "z", "bo (Hz)"))

            if rev_slice_order:
                coefs = coefs[::-1]

            cfxfull = 30082
            cfyfull = 30430
            cfzfull = 30454
            cfxfs = cfyfs = cfzfs = 5
            shim_scale = 16

            coefs[:, 1] = coefs[:, 1] / cfxfull / shim_scale * cfxfs
            coefs[:, 2] = coefs[:, 2] / cfyfull / shim_scale * cfyfs
            coefs[:, 3] = coefs[:, 3] / cfzfull / shim_scale * cfzfs

            for i_shim in range(coefs.shape[0]):
                f.write("%12s%13s%13s%13s\n" % (f"{coefs[i_shim][1]:.6f}",
                                                f"{coefs[i_shim][2]:.6f}",
                                                f"{coefs[i_shim][3]:.6f}",
                                                f"{coefs[i_shim][0]:.6f}"))


def coefs_to_dict(coefs_coil, scanner_coil_order, manufacturer):
    """ Convert the shim coefficients to a dictionary format based on the scanner coil order.

    Args:
        coefs_coil (np.array): Array of shim coefficients (n_shims x n_channels)
        scanner_coil_order (list): List of scanner coil orders (e.g. [0, 1, 2])
        manufacturer (str): Manufacturer of the scanner (e.g. 'GE', 'Siemens', 'Philips')

    Returns:
        dict: Dictionary with keys as scanner coil orders and values as the corresponding shim coefficients.
    """
    coefs_scanner = {}
    start_channel_scanner = 0
    for order in scanner_coil_order:
        end_channel_scanner_order = start_channel_scanner + channels_per_order(order, manufacturer)
        coefs_scanner[order] = coefs_coil[:, start_channel_scanner:end_channel_scanner_order]
        start_channel_scanner = end_channel_scanner_order
    coefs_coil = coefs_scanner

    return coefs_coil

b0shim_cli.add_command(gradient_realtime)
b0shim_cli.add_command(dynamic)
b0shim_cli.add_command(realtime_dynamic)
b0shim_cli.add_command(max_intensity)
b0shim_cli.add_command(add_shim_coefs)
b0shim_cli.add_command(convert_shim_coefs_format)
