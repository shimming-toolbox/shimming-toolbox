#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import os
import logging
import nibabel as nib
import numpy as np

from shimmingtoolbox.coils.create_coil_profiles import create_coil_profiles
from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_uncli
from shimmingtoolbox.utils import create_output_dir, save_nii_json, set_all_loggers
from shimmingtoolbox.masking.threshold import threshold as mask_threshold

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'fname_json', type=click.Path(exists=True), required=True,
              help="Input filename of json config file. "
                   "See the `tutorial <https://shimming-toolbox.org/en/latest/user_section/tutorials.html>`_ for more "
                   "details.")
@click.option('--relative-path', 'path_relative', type=click.Path(exists=True), required=False, default=None,
              help="Path to add before each file in the config file. This allows to have relative paths in the config "
                   "file. If this option is not specified, absolute paths must be provided in the config file.")
@click.option('--unwrapper', type=click.Choice(['prelude']), default='prelude', show_default=True,
              help="Algorithm for unwrapping")
@click.option('--threshold', type=float, required=True,
              help="Threshold for masking. Allowed range: [0, 1] where all scaled values lower than the threshold are "
                   "set to 0.")
@click.option('--autoscale-phase', 'autoscale', type=click.BOOL, default=True, show_default=True,
              help="Tells whether to auto rescale phase inputs according to manufacturer standards. If you have non "
                   "standard data, it would be preferable to set this option to False and input your phase data from "
                   "-pi to pi to avoid unwanted rescaling")
@click.option('--gaussian-filter', 'gaussian_filter', type=bool, show_default=True, help="Gaussian filter for B0 maps")
@click.option('--sigma', type=float, default=1, help="Standard deviation of gaussian filter. Used for: gaussian_filter")
@click.option('-o', '--output', 'fname_output', type=click.Path(), required=False,
              default=os.path.join(os.path.curdir, 'coil_profiles.nii.gz'),
              help="Output path filename of coil profile nifti file. Supported types : '.nii', '.nii.gz'")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def create_coil_profiles_cli(fname_json, path_relative, autoscale, unwrapper, threshold, gaussian_filter, sigma,
                             fname_output, verbose):
    """Create b0 coil profiles from acquisitions defined in the input json file"""

    # Set logger level
    set_all_loggers(verbose)

    # Config file is set up:
    #     "phase": [i_channel][i_currents][i_echo],
    #     "mag": [i_channel][i_currents][i_echo],
    #     "setup_currents": [i_channel][i_currents],
    #   "name": "Awesome_coil",
    #   "n_channels": 7,
    #   "units": "A",
    #   "coef_channel_minmax": [i_channel][2]
    #   "coef_sum_max": null

    # Get directory
    path_output = os.path.dirname(os.path.abspath(fname_output))
    create_output_dir(path_output)

    # Open json config file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    # Init variables
    phases = json_data["phase"]
    mags = json_data["mag"]
    list_setup_currents = json_data["setup_currents"]
    n_channels = json_data["n_channels"]

    # Handle relative paths in the config file
    if path_relative is not None:
        path_relative = os.path.abspath(path_relative)
        logger.info(path_relative)
        if os.path.isdir(path_relative):
            for i_channel in range(n_channels):
                n_currents = len(phases[i_channel])
                for i_current in range(n_currents):
                    n_echoes = len(phases[i_channel][i_current])
                    for i_echo in range(n_echoes):
                        # Add the relative path to the fnames in the config file
                        phases[i_channel][i_current][i_echo] = os.path.join(path_relative,
                                                                            phases[i_channel][i_current][i_echo])
                        mags[i_channel][i_current][i_echo] = os.path.join(path_relative,
                                                                          mags[i_channel][i_current][i_echo])

    # Find dead channels (mag or phase is not filled)
    dead_channels = []
    for i_channel in range(n_channels):
        if not phases[i_channel][0] or not mags[i_channel][0]:
            dead_channels.append(i_channel)
            continue

    if len(dead_channels) == n_channels:
        raise ValueError("All channels are empty. Verify input.")

    # Create a mask containing the threshold of all channels and currents
    fname_mask = os.path.join(path_output, 'mask.nii.gz')
    for i_channel in range(n_channels):
        if i_channel not in dead_channels:
            fname_mag = mags[i_channel][0][0]
            break
    nii_mag = nib.load(fname_mag)
    mask = np.full_like(nii_mag.get_fdata(), True, bool)

    for i_channel in range(n_channels):

        # If channel is dead, dont include it in the mask
        if i_channel in dead_channels:
            continue

        n_currents = len(phases[i_channel])
        channel_mask = np.full_like(nii_mag.get_fdata(), True, bool)
        for i_current in range(n_currents):
            n_echoes = len(phases[i_channel][i_current])
            # Calculate the average mag image
            current_mean = np.zeros_like(nii_mag.get_fdata())
            for i_echo in range(n_echoes):
                fname_mag = mags[i_channel][i_current][i_echo]
                mag = nib.load(fname_mag).get_fdata()
                current_mean += mag

            # Threshold mask for i_channel, i_current and all echoes
            current_mean /= n_echoes
            tmp_mask = mask_threshold(current_mean, threshold, scaled_thr=True)

            # And mask for a i_channel but all currents
            channel_mask = np.logical_and(tmp_mask, channel_mask)

        # And mask for all channels
        mask = np.logical_and(channel_mask, mask)

    # Mask contains the region where all channels get enough signal
    nii_mask = nib.Nifti1Image(mask.astype(int), nii_mag.affine, header=nii_mag.header)
    nib.save(nii_mask, fname_mask)

    if not dead_channels:
        logger.warning(f"Channels: {dead_channels} do(es) not have phase or magnitude data. They will be set to 0.")

    # Calculate fieldmaps
    fnames_fmap = []
    index_channel = -1
    for i_channel in range(n_channels):
        if i_channel in dead_channels:
            continue
        # Keeps track of the index since there could be dead channels
        index_channel += 1

        # Unwrapping each current individually leads to n*2pi offset between currents. This induces an error in the
        # linear regression in the next steps. To avoid this, we can correct the 2npi offset by assuming that the
        # difference in current does not induce more than 2pi dephasing on average in the mask (ROI). In the dataset
        # tested, the maximum offset for 0.75 amps was observed to be n=0.03. This leaves plenty of room for more
        # difference in current.
        # Note: The choice of reference fieldmap is not important since we are only interested in the slope and we
        # disregard the intercept and assume it is 0. (0 current should produce no field).
        # To implement the above mentioned method, we feed un fieldmapping pipeine the currents as if they were
        # timepoints along the 4th dimension.

        # Repeat the mask along 4th dimension
        n_currents = len(phases[i_channel])
        nii_mask_4d = nib.Nifti1Image(np.repeat(mask.astype(int)[..., np.newaxis], n_currents, -1), nii_mag.affine,
                                      header=nii_mag.header)
        fname_mask_4d = os.path.join(path_output, f"mask_4d.nii.gz")
        nib.save(nii_mask_4d, fname_mask_4d)

        # Merge the currents along the 4th dimension for the phase and magnitude of each echo
        n_echoes = len(phases[i_channel][0])
        list_fname_phase_4d = []
        for i_echo in range(n_echoes):
            phase_tmp = np.zeros(nii_mag.shape + (n_currents,))
            for i_current in range(n_currents):
                nii_phase_tmp = nib.load(phases[i_channel][i_current][i_echo])
                phase_tmp[..., i_current] = nii_phase_tmp.get_fdata()

            nii_phase_4d = nib.Nifti1Image(phase_tmp, nii_phase_tmp.affine, header=nii_phase_tmp.header)
            fname_4d = os.path.join(path_output, f"phase_channel{i_channel}_echo{i_echo}_4d.nii.gz")
            list_fname_phase_4d.append(fname_4d)
            nib.save(nii_phase_4d, fname_4d)

            # Save the json file of the 4d phase
            fname_json_4d = fname_4d.rsplit('.nii', 1)[0] + '.json'
            fname_json_3d = phases[i_channel][0][i_echo].rsplit('.nii', 1)[0] + '.json'
            # Open and save as new name
            with open(fname_json_3d) as json_file:
                json_data_3d = json.load(json_file)
            with open(fname_json_4d, mode='w') as f:
                json.dump(json_data_3d, f, indent=4)

        # Mag
        mag_tmp = np.zeros(nii_mag.shape + (n_currents,))
        for i_current in range(n_currents):
            nii_mag_tmp = nib.load(mags[i_channel][i_current][0])
            mag_tmp[..., i_current] = nii_mag_tmp.get_fdata()

        nii_mag_4d = nib.Nifti1Image(mag_tmp, nii_mag_tmp.affine, header=nii_mag_tmp.header)
        fname_4d = os.path.join(path_output, f"mag_channel{i_channel}_echo0_4d.nii.gz")
        nib.save(nii_mag_4d, fname_4d)

        # Save the json file of the 4d mag
        fname_json_4d = fname_4d.rsplit('.nii', 1)[0] + '.json'
        fname_json_3d = mags[i_channel][0][0].rsplit('.nii', 1)[0] + '.json'
        # Open and save as new name
        with open(fname_json_3d) as json_file:
            json_data_3d = json.load(json_file)
        with open(fname_json_4d, mode='w') as f:
            json.dump(json_data_3d, f, indent=4)

        fname_fmap = os.path.join(path_output, f"channel{i_channel}_fieldmap.nii.gz")
        prepare_fieldmap_uncli(list_fname_phase_4d, fname_4d, unwrapper, fname_fmap, autoscale,
                               fname_mask=fname_mask_4d,
                               gaussian_filter=gaussian_filter,
                               sigma=sigma)

        nii_fmap_4d = nib.load(fname_fmap)
        fnames_fmap.append([])
        for i_current in range(n_currents):
            fname_fmap_3d = os.path.join(path_output, f"channel{i_channel}_{i_current}_fieldmap.nii.gz")
            nii = nib.Nifti1Image(nii_fmap_4d.get_fdata()[..., i_current], nii_fmap_4d.affine,
                                  header=nii_fmap_4d.header)
            nib.save(nii, fname_fmap_3d)
            fnames_fmap[index_channel].append(fname_fmap_3d)

        # If not debug, remove junk output
        if not logger.level <= getattr(logging, 'DEBUG'):
            # Delete tmp fieldmaps
            os.remove(fname_fmap)
            fname = fname_fmap.rsplit('.nii', 1)[0] + '.json'
            os.remove(fname)

            # Delete 4d tmp phase files
            for fname in list_fname_phase_4d:
                os.remove(fname)
                fname = fname.rsplit('.nii', 1)[0] + '.json'
                os.remove(fname)

            # Delete 4d mag tmp files
            os.remove(fname_4d)
            fname = fname_4d.rsplit('.nii', 1)[0] + '.json'
            os.remove(fname)

    # Remove dead channels from the list of currents
    for i_channel in dead_channels:
        list_setup_currents.pop(i_channel)

    # Create coil profiles
    profiles = create_coil_profiles(fnames_fmap, list_currents=list_setup_currents)

    # Add dead channels as 0s
    for i_dead_channel in dead_channels:
        profiles = np.insert(profiles, i_dead_channel, np.zeros(profiles.shape[:3]), axis=3)

    # If not debug, remove junk output
    if not logger.level <= getattr(logging, 'DEBUG'):
        # Delete tmp masks
        os.remove(fname_mask)
        os.remove(fname_mask_4d)

        # For each channel
        for list_fnames in fnames_fmap:
            # For each fieldmap
            for fname_nifti in list_fnames:
                os.remove(fname_nifti)

    # Use header and json info from first file in the list
    for i_channel in range(n_channels):
        if i_channel not in dead_channels:
            fname_json_phase = phases[i_channel][0][0].rsplit('.nii', 1)[0] + '.json'
            fname_phase = phases[i_channel][0][0]
            break

    with open(fname_json_phase) as json_file:
        json_data_phase = json.load(json_file)

    nii_phase = nib.load(fname_phase)
    nii_profiles = nib.Nifti1Image(profiles, nii_phase.affine, header=nii_phase.header)

    # Save nii and json
    save_nii_json(nii_profiles, json_data_phase, fname_output)
    logger.info(f"Filename of the coil profiles is: {fname_output}")

    # Create coil config file
    coil_name = json_data['name']

    config_coil = {
        'name': coil_name,
        'coef_channel_minmax': json_data['coef_channel_minmax'],
        'coef_sum_max': json_data['coef_sum_max'],
        'Units': json_data['units']
    }

    # write json
    fname_coil_config = os.path.join(path_output, coil_name + '_config.json')
    with open(fname_coil_config, mode='w') as f:
        json.dump(config_coil, f, indent=4)

    logger.info(f"Filename of the coil config file is: {fname_coil_config}")


def _concat_and_save_nii(list_fnames_nii, fname_output):
    res = []
    for _, fname in enumerate(list_fnames_nii):
        nii = nib.load(fname)
        nii.get_fdata()
        res.append(nii.get_fdata())

    fname_json = fname.split('.nii')[0] + '.json'
    # Read from json file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    array_4d = np.moveaxis(np.array(res), 0, 3)
    nii_4d = nib.Nifti1Image(array_4d, nii.affine, header=nii.header)
    save_nii_json(nii_4d, json_data, fname_output)
