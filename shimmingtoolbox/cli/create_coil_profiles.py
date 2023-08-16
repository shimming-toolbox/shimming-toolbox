#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import os
import logging
import nibabel as nib
import numpy as np
import pathlib
import tempfile
import re

from shimmingtoolbox.coils.create_coil_profiles import create_coil_profiles, get_wire_pattern
from shimmingtoolbox.coils.biot_savart import generate_coil_bfield
from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_uncli
from shimmingtoolbox.utils import create_output_dir, save_nii_json, set_all_loggers
from shimmingtoolbox.masking.threshold import threshold as mask_threshold

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
GAMMA = 42.576E6 # in Hz/Tesla

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group(context_settings=CONTEXT_SETTINGS,
             help="Create coil profiles according to the specified algorithm as an argument e.g. st_create_coil_profiles xxxxx")
def coil_profiles_cli():
    pass

@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'fname_json', type=click.Path(exists=True), required=True,
              help="Input filename of json config file. "
                   "See the `tutorial <https://shimming-toolbox.org/en/latest/user_section/tutorials/create_b0_coil_profiles.html#create-b0-coil-profiles.html>`_ "
                   "for more details.")
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
              help="Output filename of the coil profiles NIfTI file. Supported types : '.nii', '.nii.gz'")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def from_field_maps(fname_json, path_relative, autoscale, unwrapper, threshold, gaussian_filter, sigma,
                             fname_output, verbose):
    """ Create \u0394B\u2080 coil profiles from acquisitions defined in the input json file. The output is in Hz/<current> where
        current depends on the value in the configuration file"""

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

    # Get directory and set default name
    fname_output = os.path.abspath(fname_output)
    if os.path.isdir(fname_output):
        os.path.join(fname_output, 'coil_profiles.nii.gz')
    path_output = os.path.dirname(fname_output)

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
        if not phases[i_channel] or not mags[i_channel]:
            dead_channels.append(i_channel)
            continue
        if not phases[i_channel][0] or not mags[i_channel][0]:
            dead_channels.append(i_channel)
            continue

    if len(dead_channels) == n_channels:
        raise ValueError("All channels are empty. Verify input.")

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Create a mask containing the threshold of all channels and currents
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        for i_channel in range(n_channels):
            if i_channel not in dead_channels:
                fname_mag = mags[i_channel][0][0]
                break
        nii_mag = nib.load(fname_mag)
        mask = np.full_like(nii_mag.get_fdata(), True, bool)

        for i_channel in range(n_channels):

            # If a channel is dead, don't include it in the mask
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

        if dead_channels:
            logger.warning(f"Channels: {dead_channels} do(es) not have phase or magnitude data. They will be set to 0.")

        # Calculate fieldmaps
        fnames_fmap = []
        index_channel = -1
        for i_channel in range(n_channels):
            if i_channel in dead_channels:
                continue
            # Keeps track of the index since there could be dead channels
            index_channel += 1

            # Unwrapping each current individually can lead to n*2pi offset between currents for phase difference and
            # dual-echo field mapping. This induces an error in the linear regression in the next steps. To avoid this,
            # we can correct the 2npi offset by assuming that the difference in current does not induce more than 2pi
            # dephasing on average in the mask (ROI). In the dataset tested, the maximum offset for 0.75 amps was
            # observed to be n=0.03. This leaves plenty of room for more difference in current.
            # In the multi-echo case, there are 2npi offset between echoes, these are corrected and produce a valid
            # field map. Therefore, there is no need to correct for any offset between currents.
            # Note: The choice of reference field map is not important since we are only interested in the slope and we
            # disregard the intercept.
            # To implement the above mentioned method, we feed the field mapping pipeline the currents as if they were
            # time-points along the 4th dimension.

            # TODO: Parse input and avoid recalculating the same field map if the same filename is provided more than
            #  once
            # This can only be done for multi-echo field maps since we need the unwrapped phase to correct for 2pi
            # offset in the phase difference/dual-echo case

            # Repeat the mask along 4th dimension
            n_currents = len(phases[i_channel])
            nii_mask_4d = nib.Nifti1Image(np.repeat(mask.astype(int)[..., np.newaxis], n_currents, -1), nii_mag.affine,
                                          header=nii_mag.header)
            fname_mask_4d = os.path.join(tmp, f"mask_4d.nii.gz")
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
                fname_4d = os.path.join(tmp, f"phase_channel{i_channel}_echo{i_echo}_4d.nii.gz")
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
            fname_4d = os.path.join(tmp, f"mag_channel{i_channel}_echo0_4d.nii.gz")
            nib.save(nii_mag_4d, fname_4d)

            # Save the json file of the 4d mag
            fname_json_4d = fname_4d.rsplit('.nii', 1)[0] + '.json'
            fname_json_3d = mags[i_channel][0][0].rsplit('.nii', 1)[0] + '.json'
            # Open and save as new name
            with open(fname_json_3d) as json_file:
                json_data_3d = json.load(json_file)
            with open(fname_json_4d, mode='w') as f:
                json.dump(json_data_3d, f, indent=4)

            fname_fmap = os.path.join(tmp, f"channel{i_channel}_fieldmap.nii.gz")
            prepare_fieldmap_uncli(list_fname_phase_4d, fname_4d, unwrapper, fname_fmap, autoscale,
                                   fname_mask=fname_mask_4d,
                                   gaussian_filter=gaussian_filter,
                                   sigma=sigma)

            nii_fmap_4d = nib.load(fname_fmap)
            fnames_fmap.append([])
            for i_current in range(n_currents):
                fname_fmap_3d = os.path.join(tmp, f"channel{i_channel}_{i_current}_fieldmap.nii.gz")
                nii = nib.Nifti1Image(nii_fmap_4d.get_fdata()[..., i_current], nii_fmap_4d.affine,
                                      header=nii_fmap_4d.header)
                nib.save(nii, fname_fmap_3d)
                fnames_fmap[index_channel].append(fname_fmap_3d)

        # Remove dead channels from the list of currents
        for i_channel in dead_channels:
            list_setup_currents.pop(i_channel)

        # Create coil profiles
        profiles = create_coil_profiles(fnames_fmap, list_currents=list_setup_currents)

    # Add dead channels as 0s
    for i_dead_channel in dead_channels:
        profiles = np.insert(profiles, i_dead_channel, np.zeros(profiles.shape[:3]), axis=3)

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


@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'fname_txt', type=click.Path(exists=True), required=True,
              help="Input filename of wires' geometry text file. ")
@click.option('--fmap', 'fname_fmap', required=True, type=click.Path(exists=True),
              help="Static \u0394B\u2080 fieldmap on which to calculate coil profiles. Only FOV and affine are used.")
@click.option('--offset', 'offset', required=False, type=(float, float, float), default=(0,0,0),
              help="XYZ offset: The difference between the coil’s isocenter position and the field map's isocenter position (in mm). "
                   "Input should be --offset x y z. Defaulted to 0 0 0")
@click.option('--flip', 'dims_to_flip', required=False, type=(float, float, float), default=(1,1,1),
              help="Dimensions (XYZ order) to flip in the wires' geometry (1 for no flip, -1 for flip). "
              "Input should be --flip x y z. Defaulted to 1 1 1.")
@click.option('--software', type=click.Choice(['autocad']), default='autocad',
              help=f"Software from which the geometries were extracted.")
@click.option('--coil_name', 'coil_name', required=False, type=click.STRING, default="new",
              help="Name of the coil. If not provided, \"new\" will be used.")
@click.option('--min', 'min_current', required=False, type=click.FLOAT, default=-1,
              help="The minimum current in amps going through each channel. Defaulted to -1 A.")
@click.option('--max', 'max_current', required=False, type=click.FLOAT, default=1,
              help="The maximum current in amps going through each channel. Defaulted to 1 A.")
@click.option('--max_sum', 'max_current_sum', required=False, type=click.FLOAT, default=None,
              help="The maximum sum of currents in amps going through all loops. Defaulted to the number of channel.")
@click.option('-o', '--output', 'fname_output', type=click.Path(), required=False,
              default=os.path.join(os.path.curdir, '.'),
              help="Output filename of the coil profiles NIfTI file. Supported types : '.nii', '.nii.gz'")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info', help="Be more verbose")
def from_cad(fname_txt, fname_fmap, offset, dims_to_flip, software, coil_name, min_current, max_current, max_current_sum, fname_output, verbose):
    """Create \u0394B\u2080 coil profiles from CAD wire geometries."""
    # Assert inputs
    if min_current > max_current:
        raise ValueError(f"Minimum current should be smaller than maximum current ({min_current} >= {max_current})")
    # Set logger level
    set_all_loggers(verbose)

    # create the output folder
    create_output_dir(fname_output)

    # Set variables
    nii_fmap = nib.load(fname_fmap)
    pmcn = cad_to_pumcin(fname_txt, list(dims_to_flip), software)
    pmcn[:, 1:4] += np.array(list(offset))
    wires = get_wire_pattern(pmcn)
    transform = nii_fmap.affine[:-1, :]
    nb_channels = len(wires)

    # Map the position (in mm) of all pixel in the FOV
    fov_shape = nii_fmap.header.get_data_shape()
    xx = np.arange(fov_shape[0])
    yy = np.arange(fov_shape[1])
    zz = np.arange(fov_shape[2])
    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')
    voxel_coords = np.array([x.ravel(order='F'), y.ravel(order='F'), z.ravel(order='F')])
    voxel_coords = np.vstack((voxel_coords, np.ones(voxel_coords.shape[1])))
    world_coords = transform @ voxel_coords
    gridSize = x.shape

    # Generate the coil profiles
    coil_profiles = np.zeros((gridSize[0], gridSize[1], gridSize[2], len(wires)))
    for iCh in range(len(wires)):
        coil_profiles[:, :, :, iCh] = generate_coil_bfield(wires[iCh], world_coords.T, gridSize)
    coil_profiles *= GAMMA

    # Save the coil profiles
    affine = nii_fmap.affine
    header = nii_fmap.header
    nii = nib.Nifti1Image(coil_profiles, affine=affine, header=header)
    nib.save(nii, os.path.join(fname_output, coil_name + "_coil_profiles.nii.gz"))

    # Create the coil profiles json file
    if max_current_sum is None:
        max_current_sum = nb_channels
    coef_channel_minmax = [[min_current, max_current]] * nb_channels
    config_coil = {
        'name': coil_name,
        'coef_channel_minmax': coef_channel_minmax,
        'coef_sum_max': max_current_sum,
        'Units': "A"
    }

    # Save the coil profiles json file
    fname_coil_config = os.path.join(fname_output, coil_name + '_coil_config.json')
    with open(fname_coil_config, mode='w') as f:
        json.dump(config_coil, f, indent=4)


def cad_to_pumcin(fname_txt, dimsToFlip, software):
    """Transforms CAD format to PUMCIN format"""
    # Only available txt format at the moment
    if software != "autocad":
        raise ValueError("'autocad' is the only available format at the moment")
    # TODO: Implement other software formats (SolidWorks, etc)

    _, file_ext = os.path.split(fname_txt)
    _, ext = os.path.splitext(file_ext)

    if ext != '.txt':
        raise TypeError("Geometries should be a txt file")
    with open(fname_txt, 'r') as fid:
        lines = fid.readlines()

    xyz = []
    for line in lines:
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(matches) >= 3:
            values = [float(match) for match in matches[:3]]
            xyz.append(values)

    if len(xyz) <= 0:
        raise TypeError("Data format doesn't match AutoCAD format")

    xyz = np.array(xyz)
    xyzw = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyzw[:, 0:3] = xyzw[:, 0:3] * np.array(dimsToFlip)
    xyzw[0, 3] = 0

    nPoints = xyzw.shape[0]
    iCoil = 0
    iCoilStart = [1]
    iCoilEnd = []
    TOLERANCE = 0.001

    iPoint = 0
    startPoint = xyzw[iPoint, 0:3].reshape(1, 3)

    while iPoint < nPoints:
        iPoint += 1
        distanceToStartPoint = np.linalg.norm(xyzw[iPoint, 0:3] - startPoint[iCoil, :])

        if distanceToStartPoint < TOLERANCE:
            iCoilEnd.append(iPoint)
            iCoil += 1
            iPoint += 1
            iCoilStart.append(iPoint)
            if iPoint < nPoints:
                startPoint = np.vstack((startPoint, xyzw[iPoint, 0:3].reshape(1, 3)))
                xyzw[iPoint, 3] = 0
        else:
            xyzw[iPoint, 3] = 1

    IXYZW = np.hstack((np.arange(xyzw.shape[0])[..., None], xyzw))

    return IXYZW


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


coil_profiles_cli.add_command(from_field_maps)
coil_profiles_cli.add_command(from_cad)
