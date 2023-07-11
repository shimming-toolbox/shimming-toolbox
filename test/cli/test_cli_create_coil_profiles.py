#!usr/bin/env python3
# coding: utf-8
import copy
from click.testing import CliRunner
import json
import nibabel as nib
import os
import pathlib
import pytest
import tempfile

from shimmingtoolbox.cli.create_coil_profiles import coil_profiles_cli
from shimmingtoolbox import __dir_testing__

coil_profile_config = {
    'phase': [
        [
            [
                'sub-fieldmap_phase1.nii.gz',
                'sub-fieldmap_phase2.nii.gz'
            ],
            [
                'sub-fieldmap_phase1.nii.gz',
                'sub-fieldmap_phase2.nii.gz'
            ]
        ],
    ],
    'mag': [
        [
            [
                'sub-fieldmap_magnitude1.nii.gz',
                'sub-fieldmap_magnitude2.nii.gz'
            ],
            [
                'sub-fieldmap_magnitude1.nii.gz',
                'sub-fieldmap_magnitude2.nii.gz'
            ]
        ],
        []
    ],
    "setup_currents": [
        [-0.5, 0.5],
    ],
    "name": "test_coil",
    "n_channels": 1,
    "units": "A",
    "coef_channel_minmax": [
        [-2.5, 2.5],
    ],
    "coef_sum_max": None
}


def test_create_coil_profiles():
    runner = CliRunner()

    config = copy.deepcopy(coil_profile_config)

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_config = os.path.join(tmp, 'config.json')
        with open(fname_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        fname_output = os.path.join(tmp, 'profile.nii.gz')

        res = runner.invoke(coil_profiles_cli,
                            ['from-field-maps',
                             '-i', fname_config,
                             '--relative-path', os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap'),
                             '--threshold', '0.4',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)


def test_create_coil_profiles_dead_channel1():
    runner = CliRunner()

    config = copy.deepcopy(coil_profile_config)
    config['phase'].append([])
    config['mag'].append([])
    config['setup_currents'].append([])
    config['n_channels'] = 2

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_config = os.path.join(tmp, 'config.json')
        with open(fname_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        fname_output = os.path.join(tmp, 'profile.nii.gz')

        res = runner.invoke(coil_profiles_cli,
                            ['from-field-maps',
                             '-i', fname_config,
                             '--relative-path', os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap'),
                             '--threshold', '0.4',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)
        assert nib.load(fname_output).shape == (128, 76, 10, 2)


def test_create_coil_profiles_dead_channel2():
    runner = CliRunner()

    config = copy.deepcopy(coil_profile_config)
    config['phase'].append([])
    config['phase'][1].append([])
    config['mag'].append([])
    config['setup_currents'].append([])
    config['n_channels'] = 2

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_config = os.path.join(tmp, 'config.json')
        with open(fname_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        fname_output = os.path.join(tmp, 'profile.nii.gz')

        res = runner.invoke(coil_profiles_cli,
                            ['from-field-maps',
                             '-i', fname_config,
                             '--relative-path', os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap'),
                             '--threshold', '0.4',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)
        assert nib.load(fname_output).shape == (128, 76, 10, 2)


def test_create_coil_profiles_no_channel():
    runner = CliRunner()

    config = copy.deepcopy(coil_profile_config)
    config['phase'] = []
    config['n_channels'] = 0

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        with pytest.raises(ValueError, match="All channels are empty. Verify input."):
            fname_config = os.path.join(tmp, 'config.json')
            with open(fname_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            fname_output = os.path.join(tmp, 'profile.nii.gz')

            runner.invoke(coil_profiles_cli,
                          ['from-field-maps',
                           '-i', fname_config,
                           '--relative-path', os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap'),
                           '--threshold', '0.4',
                           '-o', fname_output], catch_exceptions=False)
