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
import pickle
import numpy as np

from shimmingtoolbox.coils.coil import Coil
from shimmingtoolbox.utils import are_niis_equal, are_jsons_equal
from shimmingtoolbox.cli.create_coil_profiles import coil_profiles_cli
from shimmingtoolbox.masking.shapes import shapes
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
    "coef_channel_minmax": {"coil": [[-2.5, 2.5], ]},
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
                             '--unwrapper', 'skimage',
                             '--threshold', '0.4',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'mask.nii.gz'))


def test_integrate_coil_profile_load_constraints():
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
                             '--unwrapper', 'skimage',
                             '--threshold', '0.4',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        fname_constraints = os.path.join(tmp, 'test_coil_constraints.json')
        assert os.path.isfile(fname_constraints)
        with open(fname_constraints) as json_file:
            json_data = json.load(json_file)

        nii_profile = nib.load(fname_output)
        coil = Coil(nii_profile.get_fdata(), nii_profile.affine, json_data)
        assert coil.coef_channel_minmax == {"coil": [[-2.5, 2.5], ]}


def test_create_coil_profiles_mask():
    runner = CliRunner()

    config = copy.deepcopy(coil_profile_config)

    fname_fmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
    nii_fmap = nib.load(fname_fmap)
    nx, ny, nz = nii_fmap.shape
    mask = shapes(nii_fmap.get_fdata(), 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=int(nz / 2))

    nii_mask = nib.Nifti1Image(mask.astype(np.uint8), nii_fmap.affine)

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_config = os.path.join(tmp, 'config.json')
        with open(fname_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        fname_output = os.path.join(tmp, 'profile.nii.gz')

        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask, fname_mask)

        res = runner.invoke(coil_profiles_cli,
                            ['from-field-maps',
                             '-i', fname_config,
                             '--relative-path', os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap'),
                             '--mask', fname_mask,
                             '--unwrapper', 'skimage',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)
        assert os.path.isfile(os.path.join(tmp, 'mask.nii.gz'))


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
                             '--unwrapper', 'skimage',
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
                             '--unwrapper', 'skimage',
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


def test_create_coil_profiles_from_cad():
    runner = CliRunner()
    fname_txt = os.path.join(__dir_testing__, 'ds_coil', 'NP13_15RX_ACDC_geometries.txt')
    fname_ref_coil_profiles = os.path.join(__dir_testing__, 'ds_coil', 'NP15ch_coil_profiles.nii.gz')
    with open(os.path.join(__dir_testing__, 'ds_coil', 'NP15ch_constraints.json'), 'rb') as f:
        ref_constraints = json.load(f)
    with open(os.path.join(__dir_testing__, 'ds_coil', 'header_test.pkl'), 'rb') as outp:
        header = pickle.load(outp)
    ref_fov_shape = header["dim"][1:4]
    affine_test = np.array([header["srow_x"], header["srow_y"], header["srow_z"], [0, 0, 0, 1]])

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fm = nib.Nifti1Image(np.zeros(ref_fov_shape), affine=affine_test, header=header)
        nib.save(fm, os.path.join(tmp, 'field_map_ref.nii.gz'))
        fname_fmap = os.path.join(tmp, 'field_map_ref.nii.gz')
        fname_output = os.path.join(tmp, 'results')
        coil_name = "NP15ch"

        res = runner.invoke(coil_profiles_cli, ['from-cad',
                                                '-i', fname_txt,
                                                '--fmap', fname_fmap,
                                                '--coil_name', coil_name,
                                                '--offset', '0', '-111', '-47',
                                                '-o', fname_output],
                            catch_exceptions=False)

        assert res.exit_code == 0

        fname_cp = os.path.join(fname_output, coil_name + '_coil_profiles.nii.gz')
        assert os.path.isfile(fname_cp)

        fname_constraints = os.path.join(fname_output, coil_name + '_coil_constraints.json')
        assert os.path.isfile(fname_constraints)

        nii_test = nib.load(fname_cp)
        nii_ref = nib.load(fname_ref_coil_profiles)
        assert are_niis_equal(nii_test, nii_ref)

        with open(fname_constraints, 'rb') as f:
            config_test = json.load(f)

        assert are_jsons_equal(config_test, ref_constraints)


def test_create_coil_constraints():
    runner = CliRunner()

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_output = os.path.join(tmp, 'constraints.json')
        res = runner.invoke(coil_profiles_cli,
                            ['constraint-file',
                             '--name', 'dummy',
                             '--channels', '8',
                             '--min', '-2.5',
                             '--max', '2.5',
                             '--max-sum', '20',
                             '--units', 'A',
                             '-o', fname_output], catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)
        with open(fname_output) as json_file:
            json_data = json.load(json_file)

        expected = {"name": "dummy",
                    "Units": "A",
                    "coef_channel_minmax": {"coil": [[-2.5, 2.5], ] * 8},
                    "coef_sum_max": 20.0}
        assert json_data == expected
