#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click
import json
import os
from click.testing import CliRunner
import logging
import nibabel as nib

from shimmingtoolbox.coils.create_coil_profiles import create_coil_profiles
from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli
from shimmingtoolbox.utils import create_output_dir

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TODO
threshold = 5

@click.command(
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--input', 'fname_json', type=click.Path(), required=True,
              help="Input filename of json config file")
@click.option('-o', '--output', 'fname_output', type=click.Path(), required=False,
              default=os.path.join(os.path.curdir, 'coil_profiles.nii.gz'),
              help="Output path filename of coil profile nifti file. Supported types : '.nii', '.nii.gz'")
def create_coil_profiles_cli(fname_json, fname_output):

    # Get directory
    path_output = os.path.dirname(os.path.abspath(fname_output))
    create_output_dir(path_output)

    # Open json config file
    with open(fname_json) as json_file:
        json_data = json.load(json_file)

    phases = json_data["phases"]
    mags = json_data["mag"]

    list_diff = json_data["diff"]

    n_channels = len(phases)
    min_max_fmaps = []
    for i_channel in range(n_channels):
        min_phases = phases[i_channel][0]
        max_phases = phases[i_channel][1]

        # TODO: Change mag for each unwrap? channel?
        fname_mag = mags[i_channel][0][0]
        n_echoes = len(min_phases)

        runner = CliRunner()
        fname_min_output = os.path.join(path_output, f"channel{i_channel}_min_fieldmap.nii.gz")
        result = runner.invoke(prepare_fieldmap_cli, [*min_phases,
                                                      '--mag', fname_mag,
                                                      '--unwrapper', 'prelude',
                                                      '--threshold', threshold,
                                                      '-o', fname_min_output], catch_exceptions=False)
        if result.exit_code != 0:
            raise RuntimeError("prepare fieldmap did not run")

        runner = CliRunner()
        fname_max_output = os.path.join(path_output, f"channel{i_channel}_max_fieldmap.nii.gz")
        result = runner.invoke(prepare_fieldmap_cli, [*max_phases,
                                                      '--mag', fname_mag,
                                                      '--unwrapper', 'prelude',
                                                      '--threshold', threshold,
                                                      '-o', fname_max_output], catch_exceptions=False)
        if result.exit_code != 0:
            raise RuntimeError("prepare fieldmap did not run")

        min_max_fmaps.append([fname_min_output, fname_max_output])

    profiles = create_coil_profiles(min_max_fmaps, list_diff=list_diff)

    nii = nib.load(min_phases[0])

    nii_profiles = nib.Nifti1Image(profiles, nii.affine, header=nii.header)
    nib.save(nii_profiles, fname_output)
