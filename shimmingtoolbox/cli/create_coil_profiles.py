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
              help="Input filename of json config file")
@click.option('--unwrapper', type=click.Choice(['prelude']), default='prelude', show_default=True,
              help="Algorithm for unwrapping")
@click.option('--threshold', type=float, help="Threshold for masking.")
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
def create_coil_profiles_cli(fname_json, autoscale, unwrapper, threshold, gaussian_filter, sigma, fname_output,
                             verbose):
    """Create b0 coil profiles from acquisitions defined in the input json file"""

    # Set logger level
    set_all_loggers(verbose)

    # Config file is set up:
    # phase[i_channel][min_max][i_echo]
    # mag[i_channel][min_max][i_echo]
    #     "setup_currents": [
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5]
    #   ],
    #   "name": "greg_coil",
    #   "n_channels": 7,
    #   "units": "A",
    #   "coef_channel_minmax": [i_channel][min_max]
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
    n_channels = len(phases)
    n_currents = len(phases[0])
    n_echoes = len(phases[0][0])

    # Create a mask containing the threshold of all channels and currents
    fname_mask = os.path.join(path_output, 'mask.nii.gz')
    fname_mag = mags[0][0][0]
    nii_mag = nib.load(fname_mag)
    mask = np.full_like(nii_mag.get_fdata(), True, bool)
    dead_channels = []
    for i_channel in range(n_channels):
        if not phases[i_channel][0]:
            dead_channels.append(i_channel)
            continue

        channel_mask = np.full_like(nii_mag.get_fdata(), True, bool)
        for i_current in range(n_currents):

            # Calculate the average mag image
            current_mean = np.zeros_like(nii_mag.get_fdata())
            for i_echo in range(n_echoes):
                fname_mag = mags[i_channel][i_current][i_echo]
                mag = nib.load(fname_mag).get_fdata()
                current_mean += mag

            # Threshold mask for i_channel, i_current and all echoes
            current_mean /= n_echoes
            tmp_mask = mask_threshold(current_mean, threshold)

            # And mask for a i_channel but all currents
            channel_mask = np.logical_and(tmp_mask, channel_mask)

        # And mask for all channels
        mask = np.logical_and(channel_mask, mask)

    # Mask contains the region where all channels get enough signal
    nii_mask = nib.Nifti1Image(mask.astype(int), nii_mag.affine, header=nii_mag.header)
    nib.save(nii_mask, fname_mask)

    if not dead_channels:
        logger.warning(f"Channels: {dead_channels} do not have phase data. They will be set to 0.")

    fnames_fmap = []
    index_channel = -1
    # For each channel
    for i_channel in range(n_channels):
        if i_channel in dead_channels:
            continue

        # Keeps track of the index since there could be dead channels
        index_channel += 1

        # for each current
        fnames_fmap.append([])
        for i_current in range(n_currents):
            phase = phases[i_channel][i_current]

            # Calculate fieldmap and save to a file
            fname_fmap = os.path.join(path_output, f"channel{i_channel}_{i_current}_fieldmap.nii.gz")
            prepare_fieldmap_uncli(phase, fname_mag, unwrapper, fname_fmap, autoscale,
                                   fname_mask=fname_mask,
                                   gaussian_filter=gaussian_filter,
                                   sigma=sigma)

            fnames_fmap[index_channel].append(fname_fmap)

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
        os.remove(fname_mask)
        # For each channel
        for list_fnames in fnames_fmap:
            # For each fieldmap
            for fname_nifti in list_fnames:
                fname_json_m = fname_nifti.rsplit('.nii', 1)[0] + '.json'
                # Delete nifti
                os.remove(fname_nifti)
                # Delete json
                os.remove(fname_json_m)

    # Use header and json info from first file in the list
    fname_json_phase = phases[0][0][0].rsplit('.nii', 1)[0] + '.json'
    nii_phase = nib.load(phases[0][0][0])
    nii_profiles = nib.Nifti1Image(profiles, nii_phase.affine, header=nii_phase.header)

    # Save nii and json
    save_nii_json(nii_profiles, fname_json_phase, fname_output)
    logger.info(f"\n\n Filename of the coil profiles is: {fname_output}")

    # Create coil config file
    coil_name = json_data['name']

    config_coil = {
        'name': coil_name,
        'coef_channel_minmax': json_data['coef_channel_minmax'],
        'coef_sum_max': json_data['coef_sum_max']
    }

    # write json
    fname_coil_config = os.path.join(path_output, coil_name + '_config.json')
    with open(fname_coil_config, mode='w') as f:
        json.dump(config_coil, f, indent=4)

    logger.info(f"Filename of the coil config file is: {fname_coil_config}")
