#!/usr/bin/python3
# -*- coding: utf-8 -*

import copy
import pytest
from click.testing import CliRunner
import tempfile
import pathlib
import os
import nibabel as nib
import numpy as np
import json

from shimmingtoolbox import __config_custom_coil_constraints__, __config_scanner_constraints__
from shimmingtoolbox.cli.b0shim import define_slices_cli
from shimmingtoolbox.cli.b0shim import b0shim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.coils.spher_harm_basis import siemens_basis
from shimmingtoolbox.coils.coordinates import generate_meshgrid


def _define_inputs(fmap_dim, manufacturers_model_name=None, no_shim_settings=False):
    # fname for fmap
    fname_fmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
    nii = nib.load(fname_fmap)
    fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.json')

    with open(fname_json) as f:
        fm_data = json.load(f)

    if manufacturers_model_name is not None:
        fm_data['ManufacturersModelName'] = manufacturers_model_name

    if no_shim_settings:
        fm_data['ShimSetting'] = [None]

    if fmap_dim == 4:
        nii_fmap = nii
    elif fmap_dim == 3:
        nii_fmap = nib.Nifti1Image(np.mean(nii.get_fdata(), axis=3), nii.affine, header=nii.header)
    elif fmap_dim == 2:
        nii_fmap = nib.Nifti1Image(nii.get_fdata()[..., 0, 0], nii.affine, header=nii.header)
    else:
        raise ValueError("Supported Dimensions are 2, 3 or 4")

    # fname for anat
    fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')

    nii_anat = nib.load(fname_anat)

    fname_anat_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.json')
    with open(fname_anat_json) as f:
        anat_data = json.load(f)

    anat_data['ScanOptions'] = ['FS']

    anat = nii_anat.get_fdata()

    # Set up mask: Cube
    # static
    nx, ny, nz = anat.shape
    mask = shapes(anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)

    nii_mask = nib.Nifti1Image(mask.astype(np.uint8), nii_anat.affine)

    return nii_fmap, nii_anat, nii_mask, fm_data, anat_data


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask,fm_data,anat_data", [(
            _define_inputs(fmap_dim=3)
    )]
)
class TestCliDynamic(object):
    def test_cli_dynamic_sph(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()

            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1,2',
                                             '--regularization-factor', '0.1',
                                             '--slices', 'ascending',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))
            with open(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"), 'r') as file:
                lines = file.readlines()
                line = lines[8].strip().split(',')
                values = [float(val) for val in line if val.strip()]

            assert values == [0.002985, -14.587414, -57.016499, -2.745062, -0.401786, -3.580623, 0.668977, -0.105560]

    def test_cli_dynamic_external_scanner_constraint(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            with open(__config_scanner_constraints__) as f:
                constraints_data = json.load(f)

            constraints_data['coefs_used']['0'] = [123100100 + 1000]
            constraints_data['coefs_used']['1'] = None
            constraints_data['coefs_used']['2'] = [None, 1000, 1000, 1000, 1000]
            constraints_data['coef_channel_minmax']['2'][0] = [None, None]
            constraints_data['coef_channel_minmax']['2'][1] = None
            fname_scanner_constraints_json = os.path.join(tmp, 'scanner_constraints.json')
            with open(fname_scanner_constraints_json, 'w', encoding='utf-8') as f:
                json.dump(constraints_data, f, indent=4)

            runner = CliRunner()

            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0,1,2,3',
                                             '--scanner-coil-constraints', fname_scanner_constraints_json,
                                             '--slices', 'volume',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))
            with open(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"), 'r') as file:
                lines = file.readlines()
                line = lines[8].strip().split(',')
                values = [float(val) for val in line if val.strip()]

            assert values == [-16.417611, 1.283857, -14.424815, -84.402628, -6.60401, -0.653534, -6.75787,
                              0.955701, -0.168711, -0.139256, -0.094325,  -0.798893, 0.322054]
            fname_bids_sidecar_fmap_output = os.path.join(tmp, "fieldmap_calculated_shim.json")
            assert os.path.isfile(fname_bids_sidecar_fmap_output)
            with open(fname_bids_sidecar_fmap_output) as f:
                bids_sidecar_fmap_output_data = json.load(f)
            assert bids_sidecar_fmap_output_data['ImagingFrequency'] == 123.101083
            # Siemens scanner need to be scaled to DAC units. Moreover, we input the 'coefs_used' as ui units.
            # This is why these values are not the direct sum of coefs + coefs_used
            assert bids_sidecar_fmap_output_data['ShimSetting'] == [673.058, -7405.47, -9951.41,
                                                                    None, 2813.48, 2834.6, 2818.01, 2866.49,
                                                                    None, None, None, None]

    def test_cli_dynamic_sph_table_not_iso(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fm_data = copy.deepcopy(fm_data)
            fm_data['TablePosition'] = [0, 0, 10]
            anat_data = copy.deepcopy(anat_data)
            anat_data['TablePosition'] = [0, 0, 10]
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()

            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1,2',
                                             '--regularization-factor', '0.1',
                                             '--slices', 'ascending',
                                             '--optimizer-method', 'pseudo_inverse',
                                             '--output', tmp,
                                             '--verbose', 'debug'],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))
            with open(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"), 'r') as file:
                lines = file.readlines()
                line = lines[8].strip().split(',')
                values = [float(val) for val in line if val.strip()]

            assert values == [-4.305412, -67.945413, 29.093204, -3804.340894,
                              -163.848482, -863.019747, 576.108918, -177.97076]

    def test_cli_dynamic_signal_recovery(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""

        # Duplicate nii_fmap's third dimension
        fmap = nii_fmap.get_fdata()
        fmap = np.repeat(fmap, 5, axis=2)
        nii_fmap = nib.Nifti1Image(fmap, nii_fmap.affine, header=nii_fmap.header)

        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()

            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1,2',
                                             '--regularization-factor', '0.3',
                                             '--slices', 'ascending',
                                             '--optimizer-method', 'least_squares',
                                             '--optimizer-criteria', 'grad',
                                             '--weighting-signal-loss', '0.01',
                                             '--mask-dilation-kernel-size', '5',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0

    def test_cli_dynamic_no_mask(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--scanner-coil-order', '0,1',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))

    def test_cli_dynamic_coils(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with input coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--output', tmp,
                                             '--optimizer-method', 'least_squares',
                                             '--optimizer-criteria', 'mse'],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Dummy_coil.txt"))

    def test_cli_dynamic_sph_order_0(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 0 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))

    def test_cli_dynamic_sph_order_3(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 3 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '3',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))

    def test_cli_dynamic_coils_and_sph(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with input coil and scanner coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Dummy_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil1_Prisma_fit.txt"))

    def test_cli_dynamic_format_chronological_coil(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with chronological-coil o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'chronological-coil',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))

    def test_cli_dynamic_fatsat(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with input coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--output-file-format-coil', 'chronological-coil',
                                             '--fatsat', 'no',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Dummy_coil.txt"))

    def test_cli_dynamic_format_chronological_ch(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with chronological_ch o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0,1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'chronological-ch',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))
            # There should be 4 x 10 x 1 value

    def test_cli_dynamic_format_slicewise_ch(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with slicewise_ch o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0,1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'slicewise-ch',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))

    def test_cli_dynamic_format_gradient_order01(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with gradient o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0,1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            with open(os.path.join(tmp, "f0shim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 11.007908"
            with open(os.path.join(tmp, "xshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.001260"
            with open(os.path.join(tmp, "yshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.029665"
            with open(os.path.join(tmp, "zshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.060548"


    def test_cli_dynamic_format_gradient_and_custom_coil(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with gradient o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0,1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Dummy_coil.txt"))

    def test_cli_dynamic_format_gradient_order0(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with gradient o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            with open(os.path.join(tmp, "f0shim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 119.644382"

    def test_cli_dynamic_format_gradient_order1(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil with gradient o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1',
                                             '--slice-factor', '2',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            with open(os.path.join(tmp, "xshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.001980"
            with open(os.path.join(tmp, "yshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.032016"
            with open(os.path.join(tmp, "zshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.066749"

    def test_cli_dynamic_debug_verbose(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '1',
                                             '-v', 'debug',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))
            # Artefacts of the debug verbose
            assert os.path.isfile(os.path.join(tmp, "tmp_extended_fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "anat.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "mask.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fig_currents.png"))

    def test_cli_dynamic_absolute(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask', fname_mask,
                                             '--scanner-coil-order', '0, 2',
                                             '--output-value-format', 'absolute',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"))
            with open(os.path.join(tmp, "coefs_coil0_Prisma_fit.txt"), 'r') as file:
                lines = file.readlines()
                line = lines[8].strip().split(',')
                values = [float(val) for val in line if val.strip()]

            assert values == [123259067.330864, -718.069583, 138.656751, -110.517759, 24.97596, -4.888655]

    def test_cli_2d_fmap(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):

        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            nii_fmap = nib.Nifti1Image(nii_fmap.get_fdata()[..., 0], nii_fmap.affine, header=nii_fmap.header)
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['dynamic',
                                             '--fmap', fname_fmap,
                                             '--mask', fname_mask,
                                             '--anat', fname_anat,
                                             '--scanner-coil-order', '1',
                                             '--output', tmp],
                                catch_exceptions=False)
            assert res.exit_code == 0

    def test_cli_dynamic_no_coil(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()

            with pytest.raises(RuntimeError, match="No custom or scanner coils were selected."):
                runner.invoke(b0shim_cli, ['dynamic',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--output', tmp],
                              catch_exceptions=False)

    def test_cli_dynamic_wrong_dim_info(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            nii_new_anat = copy.deepcopy(nii_anat)
            nii_new_anat.header.set_dim_info(2, 1, 0)

            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_new_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()
            with pytest.raises(RuntimeError,
                               match="Slice encode direction must be the 3rd dimension of the NIfTI file."):
                runner.invoke(b0shim_cli, ['dynamic',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                              catch_exceptions=False)

    def test_cli_dynamic_no_fmap_json(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            runner = CliRunner()

            with pytest.raises(OSError, match="Missing fieldmap json file"):
                runner.invoke(b0shim_cli, ['dynamic',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                              catch_exceptions=False)


def test_cli_dynamic_unknown_scanner():
    """Test cli with scanner coil profiles of order 1 with default constraints"""
    nii_fmap, nii_anat, nii_mask, fm_data, anat_data = _define_inputs(fmap_dim=3,
                                                                      manufacturers_model_name='not_set',
                                                                      no_shim_settings=True)
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Save the inputs to the new directory
        fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
        fname_fm_json = os.path.join(tmp, 'fmap.json')
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        fname_anat = os.path.join(tmp, 'anat.nii.gz')
        fname_anat_json = os.path.join(tmp, 'anat.json')
        _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                     nii_anat=nii_anat, fname_anat=fname_anat,
                     nii_mask=nii_mask, fname_mask=fname_mask,
                     fm_data=fm_data, fname_fm_json=fname_fm_json,
                     anat_data=anat_data, fname_anat_json=fname_anat_json)

        runner = CliRunner()

        res = runner.invoke(b0shim_cli, ['dynamic',
                                         '--fmap', fname_fmap,
                                         '--anat', fname_anat,
                                         '--mask', fname_mask,
                                         '--scanner-coil-order', '1,2',
                                         '--output', tmp],
                            catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(os.path.join(tmp, "coefs_coil0_Unknown.txt"))


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask,fm_data,anat_data", [(
            _define_inputs(fmap_dim=4)
    )]
)
class TestCLIRealtime(object):
    def test_cli_rt_sph(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--output', tmp,
                                             '-v', 'debug'],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))
            with open(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"), 'r') as file:
                lines = file.readlines()
                line = lines[5].strip().split(',')
                values = [float(val) for val in line if val.strip()]
            assert values == [11.007908, -0.014577058094, 1311.6784]

    def test_cli_rt_sph_order_0(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))

    def test_cli_rt_sph_order_02(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,2',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch4_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch5_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch6_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch7_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch8_Prisma_fit.txt"))

    def test_cli_rt_custom_coil(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--coil-riro', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--optimizer-method', 'pseudo_inverse',
                                             '--slice-factor', '2',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            for i_channel in range(9):
                assert os.path.isfile(os.path.join(tmp, f"coefs_coil0_ch{i_channel}_Dummy_coil.txt"))

    def test_cli_rt_debug(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '-v', 'debug',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))
            # Artefacts of the debug verbose
            assert os.path.isfile(os.path.join(tmp, "tmp_extended_fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "anat.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "mask.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fig_currents.png"))

    def test_cli_rt_no_mask(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))

    def test_cli_rt_chronological_ch(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--output-file-format-scanner', 'chronological-ch',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))

    def test_cli_rt_gradient_order01(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with (tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp):
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            with open(os.path.join(tmp, "f0shim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 11.007908" and lines[16].strip() == "corr_vec[1][5]= -0.014577058094"
            with open(os.path.join(tmp, "xshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.001260" and lines[16].strip() == "corr_vec[1][5]= -0.000000000000"
            with open(os.path.join(tmp, "yshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.029665" and lines[16].strip() == "corr_vec[1][5]= 0.000005532449"
            with open(os.path.join(tmp, "zshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.060548" and lines[16].strip() == "corr_vec[1][5]= 0.000013492875"

    def test_cli_rt_gradient_order1(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with (tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp):
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '1',
                                             '--scanner-coil-order-riro', '1',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            with open(os.path.join(tmp, "xshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.001980" and lines[16].strip() == ("corr_vec[1][5]= "
                                                                                                 "-0.000001091261")
            with open(os.path.join(tmp, "yshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.032016" and lines[16].strip() == ("corr_vec[1][5]= "
                                                                                                 "0.000004151859")
            with open(os.path.join(tmp, "zshim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 0.066749" and lines[16].strip() == ("corr_vec[1][5]= "
                                                                                                 "0.000003690757")

    def test_cli_rt_gradient_order0(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with (tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp):
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0',
                                             '--scanner-coil-order-riro', '0',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            with open(os.path.join(tmp, "f0shim_gradients.txt"), 'r') as file:
                lines = file.readlines()
                assert lines[15].strip() == "corr_vec[0][5]= 119.644383" and lines[16].strip() == ("corr_vec[1][5]= "
                                                                                                 "0.009161744842")

    def test_cli_rt_gradient_order01_custom_coil(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with (tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp):
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            # Dummy coil
            nii_dummy_coil, dummy_coil_constraints = _create_dummy_coil(nii_fmap)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            # Save json
            fname_constraints = os.path.join(tmp, 'dummy_coil.json')
            with open(fname_constraints, 'w', encoding='utf-8') as f:
                json.dump(dummy_coil_constraints, f, indent=4)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--coil', fname_dummy_coil, fname_constraints,
                                             '--coil-riro', fname_dummy_coil, fname_constraints,
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--optimizer-method', 'pseudo_inverse',
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--scanner-coil-order-riro', '0,1',
                                             '--output-file-format-scanner', 'gradient',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "f0shim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "xshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "yshim_gradients.txt"))
            assert os.path.isfile(os.path.join(tmp, "zshim_gradients.txt"))
            for i_channel in range(9):
                assert os.path.isfile(os.path.join(tmp, f"coefs_coil0_ch{i_channel}_Dummy_coil.txt"))


    def test_cli_rt_absolute(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1,2',
                                             '--output-value-format', 'absolute',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))

    def test_cli_rt_pseudo_inverse(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['realtime-dynamic',
                                             '--fmap', fname_fmap,
                                             '--anat', fname_anat,
                                             '--mask-static', fname_mask,
                                             '--mask-riro', fname_mask,
                                             '--resp', fname_resp,
                                             '--slice-factor', '2',
                                             '--scanner-coil-order', '0,1',
                                             '--optimizer-method', 'pseudo_inverse',
                                             '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_Prisma_fit.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_Prisma_fit.txt"))

    def test_cli_realtime_wrong_dim_info(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            nii_new_anat = copy.deepcopy(nii_anat)
            nii_new_anat.header.set_dim_info(2, 1, 0)

            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_fm_json = os.path.join(tmp, 'fmap.json')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_new_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         fm_data=fm_data, fname_fm_json=fname_fm_json,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()
            with pytest.raises(RuntimeError,
                               match="Slice encode direction must be the 3rd dimension of the NIfTI file."):
                runner.invoke(b0shim_cli, ['realtime-dynamic',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask-static', fname_mask,
                                           '--mask-riro', fname_mask,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                              catch_exceptions=False)

    def test_cli_realtime_no_fmap_json(self, nii_fmap, nii_anat, nii_mask, fm_data, anat_data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the inputs to the new directory
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            fname_anat_json = os.path.join(tmp, 'anat.json')
            _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                         nii_anat=nii_anat, fname_anat=fname_anat,
                         nii_mask=nii_mask, fname_mask=fname_mask,
                         anat_data=anat_data, fname_anat_json=fname_anat_json)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime',
                                      'sub-realtime_PMUresp_signal.resp')

            runner = CliRunner()

            with pytest.raises(OSError, match="Missing fieldmap json file"):
                runner.invoke(b0shim_cli, ['realtime-dynamic',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask-static', fname_mask,
                                           '--mask-riro', fname_mask,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                              catch_exceptions=False)


def test_cli_define_slices_def():
    """Test using a number for the number of slices"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'slices.json')
        res = runner.invoke(define_slices_cli, ['--slices', '12',
                                                '--factor', '6',
                                                '--method', 'ascending',
                                                '-o', fname_output],
                            catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)


def test_cli_define_slices_anat():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
        fname_output = os.path.join(tmp, 'slices.json')
        res = runner.invoke(define_slices_cli, ['--slices', fname_anat,
                                                '--factor', '5',
                                                '--method', 'ascending',
                                                '-o', fname_output],
                            catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)


def test_cli_define_slices_wrong_input():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_anat = os.path.join('abc.nii')
        fname_output = os.path.join(tmp, 'slices.json')
        with pytest.raises(ValueError, match="Could not get the number of slices"):
            runner.invoke(define_slices_cli, ['--slices', fname_anat,
                                              '--factor', '5',
                                              '--method', 'ascending',
                                              '-o', fname_output],
                          catch_exceptions=False)


def test_cli_define_slices_wrong_output():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'slices')
        with pytest.raises(ValueError, match="Filename of the output must be a json file"):
            runner.invoke(define_slices_cli, ['--slices', "10",
                                              '--factor', '5',
                                              '--method', 'ascending',
                                              '-o', fname_output],
                          catch_exceptions=False)


def _save_inputs(nii_fmap=None, fname_fmap=None,
                 nii_anat=None, fname_anat=None,
                 nii_mask=None, fname_mask=None,
                 fm_data=None, fname_fm_json=None,
                 anat_data=None, fname_anat_json=None):
    """Save inputs if they are not None, use the respective fnames for the different inputs to save"""
    if nii_fmap is not None:
        # Save the fieldmap
        nib.save(nii_fmap, fname_fmap)

    if fm_data is not None:
        # Save json
        with open(fname_fm_json, 'w', encoding='utf-8') as f:
            json.dump(fm_data, f, indent=4)

    if nii_anat is not None:
        # Save the anat
        nib.save(nii_anat, fname_anat)

    if anat_data is not None:
        # Save json
        with open(fname_anat_json, 'w', encoding='utf-8') as f:
            json.dump(anat_data, f, indent=4)

    if nii_mask is not None:
        # Save the mask
        nib.save(nii_mask, fname_mask)


def _create_dummy_coil(nii_fmap):
    """Create coil profiles and constraints following sph harmonics 0, 1, 2 order. This is useful for testing custom
    coils
    """
    shape = list(nii_fmap.shape[:3])
    shape[2] += 5
    mesh_x, mesh_y, mesh_z = generate_meshgrid(shape, nii_fmap.affine)
    profiles = siemens_basis(mesh_x, mesh_y, mesh_z)
    profile_order_0 = np.ones(shape)
    sph_coil_profile = np.concatenate((profile_order_0[..., np.newaxis], profiles), axis=3)
    nii_dummy_coil = nib.Nifti1Image(sph_coil_profile, nii_fmap.affine, header=nii_fmap.header)

    # Dummy constraints
    with open(__config_custom_coil_constraints__, 'r', encoding='utf-8') as f:
        constraints = json.load(f)

    constraints['name'] = 'Dummy_coil'
    return nii_dummy_coil, constraints


def test_b0_max_intensity():
    """ We use a 4d fieldmap not intended for this application for testing """
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_input = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        fname_output = os.path.join(tmp, 'output.txt')

        nii = nib.load(fname_input)
        # Set up mask: Cube
        nx, ny, nz = nii.shape[:3]
        mask = shapes(nii.get_fdata()[..., 0], 'cube',
                      center_dim1=32,
                      center_dim2=36,
                      len_dim1=10, len_dim2=10, len_dim3=nz)
        nii_mask = nib.Nifti1Image(mask.astype(np.uint8), nii.affine)
        nib.save(nii_mask, fname_mask)

        runner = CliRunner()
        res = runner.invoke(b0shim_cli, ['max-intensity',
                                         '--input', fname_input,
                                         '--mask', fname_mask,
                                         '-o', fname_output],

                            catch_exceptions=False)

        assert res.exit_code == 0
        with open(fname_output, 'r', encoding='utf-8') as f:
            assert f.readline().strip() == "1"
            assert f.readline().strip() == "9"


def test_b0_max_intensity_no_mask():
    """ We use a 4d fieldmap not intended for this application for testing """
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        fname_input = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
        fname_output = os.path.join(tmp, 'output.txt')

        runner = CliRunner()
        res = runner.invoke(b0shim_cli, ['max-intensity',
                                         '--input', fname_input,
                                         '-o', fname_output],

                            catch_exceptions=False)

        assert res.exit_code == 0
        with open(fname_output, 'r', encoding='utf-8') as f:
            assert f.readline().strip() == "1"
            assert f.readline().strip() == "1"


class TestAddShimCoefs:
    def test_add_shim_coefs(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input1 = os.path.join(tmp, 'shim_coefs1_vol.txt')
            with open(fname_input1, 'w', encoding='utf-8') as f:
                f.write("11, 12, 13, 14, 15, 11, 12, 13, 14\n")
                f.write("11, 12, 13, 14, 15, 11, 12, 13, 14,\n")

            fname_input2 = os.path.join(tmp, 'shim_coefs2.txt')
            with open(fname_input2, 'w', encoding='utf-8') as f:
                f.write("10, 10, 10, 10, 10, 10, 10, 10, 10\n")
                f.write("11, 12, 13, 14, 15, 11, 12, 13, 14,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['add-shim-coefs',
                                             '--input', fname_input1,
                                             '--input2', fname_input2,
                                             '-o', fname_output,
                                             '-v', 'debug'],
                                catch_exceptions=False)

            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == ("21.000000, 22.000000, 23.000000, 24.000000, 25.000000, "
                                        "21.000000, 22.000000, 23.000000, 24.000000,\n")
                assert f.readline() == ("22.000000, 24.000000, 26.000000, 28.000000, 30.000000, "
                                        "22.000000, 24.000000, 26.000000, 28.000000,\n")

    def test_add_shim_coefs_error(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input1 = os.path.join(tmp, 'shim_coefs1_vol.txt')
            with open(fname_input1, 'w', encoding='utf-8') as f:
                f.write("11, 12, 13, 14, 15, 11, 12, 13, 14\n")
                f.write("11, 12, 13, 14, 15, 11, 12, 13, 14,\n")

            fname_input2 = os.path.join(tmp, 'shim_coefs2.txt')
            with open(fname_input2, 'w', encoding='utf-8') as f:
                f.write("1,2,3, 4,\n")
                f.write("5,6,7, 4\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')

            runner = CliRunner()
            with pytest.raises(ValueError, match="The number of shim events and/or the number of channels is not "
                                                 "the same in both text files"):
                runner.invoke(b0shim_cli, ['add-shim-coefs',
                                                 '--input', fname_input1,
                                                 '--input2', fname_input2,
                                                 '-o', fname_output,
                                                 '-v', 'debug'],
                                    catch_exceptions=False)


class TestConvertShimCoefsFormat:
    def test_convert_shim_coefs_vol_sl(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input = os.path.join(tmp, 'shim_coefs.txt')
            with open(fname_input, 'w', encoding='utf-8') as f:
                f.write("1,2,3,4,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')
            fname_anat = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap",
                                      "sub-1_acq-gre_magnitude1.nii.gz")
            fname_json = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap", "sub-1_acq-gre_magnitude1.json")
            nii = nib.load(fname_anat)
            with open(fname_json) as f:
                json_data = json.load(f)
            json_data['SliceTiming'] = [1, 0, 2]
            fname_target = os.path.join(tmp, 'target.nii.gz')
            nib.save(nib.Nifti1Image(nii.get_fdata()[:, :, :3], nii.affine, nii.header), fname_target)
            with open(os.path.join(tmp, 'target.json'), 'w') as f:
                json.dump(json_data, f)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['convert-shim-coefs-format',
                                             '--target', fname_target,
                                             '--input', fname_input,
                                             '--input-file-format', 'volume',
                                             '--output-file-format', 'slicewise',
                                             '--add-channels', '0,4,6,7,8,9',
                                             '-o', fname_output,
                                             '-v', 'debug'],
                                catch_exceptions=False)
            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == ("0.000000, 1.000000, 2.000000, 3.000000, 0.000000, 4.000000, 0.000000, "
                                        "0.000000, 0.000000, 0.000000,\n")
                assert f.readline() == ("0.000000, 1.000000, 2.000000, 3.000000, 0.000000, 4.000000, 0.000000, "
                                        "0.000000, 0.000000, 0.000000,\n")
                assert f.readline() == ("0.000000, 1.000000, 2.000000, 3.000000, 0.000000, 4.000000, 0.000000, "
                                        "0.000000, 0.000000, 0.000000,\n")

    def test_convert_shim_coefs_ch_vol(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input = os.path.join(tmp, 'shim_coefs.txt')
            with open(fname_input, 'w', encoding='utf-8') as f:
                f.write("0,0,0,0,\n")
                f.write("0,0,0,0,\n")
                f.write("0,0,0,0,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')
            fname_anat = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap",
                                      "sub-1_acq-gre_magnitude1.nii.gz")
            fname_json = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap", "sub-1_acq-gre_magnitude1.json")
            nii = nib.load(fname_anat)
            with open(fname_json) as f:
                json_data = json.load(f)
            json_data['SliceTiming'] = [1, 0, 2]
            fname_target = os.path.join(tmp, 'target.nii.gz')
            nib.save(nib.Nifti1Image(nii.get_fdata()[:, :, :3], nii.affine, nii.header), fname_target)
            with open(os.path.join(tmp, 'target.json'), 'w') as f:
                json.dump(json_data, f)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['convert-shim-coefs-format',
                                             '--target', fname_target,
                                             '--input', fname_input,
                                             '--input-file-format', 'chronological',
                                             '--output-file-format', 'volume',
                                             '-o', fname_output,
                                             '-v', 'debug'],
                                catch_exceptions=False)
            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == "0.000000, 0.000000, 0.000000, 0.000000,"

    def test_convert_shim_coefs_ch_sl(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input = os.path.join(tmp, 'shim_coefs.txt')
            with open(fname_input, 'w', encoding='utf-8') as f:
                f.write("1, 1, 1, 1\n")
                f.write("0,0,0,0,\n")
                f.write("2, 2, 2, 2,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')
            fname_anat = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap",
                                      "sub-1_acq-gre_magnitude1.nii.gz")
            fname_json = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap", "sub-1_acq-gre_magnitude1.json")
            nii = nib.load(fname_anat)
            with open(fname_json) as f:
                json_data = json.load(f)
            json_data['SliceTiming'] = [1, 0, 2]
            fname_target = os.path.join(tmp, 'target.nii.gz')
            nib.save(nib.Nifti1Image(nii.get_fdata()[:, :, :3], nii.affine, nii.header), fname_target)
            with open(os.path.join(tmp, 'target.json'), 'w') as f:
                json.dump(json_data, f)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['convert-shim-coefs-format',
                                             '--target', fname_target,
                                             '--input', fname_input,
                                             '--input-file-format', 'chronological',
                                             '--output-file-format', 'slicewise',
                                             '-o', fname_output,
                                             '-v', 'debug'],

                                catch_exceptions=False)
            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == "0.000000, 0.000000, 0.000000, 0.000000,\n"
                assert f.readline() == "1.000000, 1.000000, 1.000000, 1.000000,\n"
                assert f.readline() == "2.000000, 2.000000, 2.000000, 2.000000,\n"

    def test_convert_shim_coefs_sl_ch(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input = os.path.join(tmp, 'shim_coefs.txt')
            with open(fname_input, 'w', encoding='utf-8') as f:
                f.write("0,0,0,0,\n")
                f.write("1, 1, 1, 1\n")
                f.write("2, 2, 2, 2,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')
            fname_anat = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap",
                                      "sub-1_acq-gre_magnitude1.nii.gz")
            fname_json = os.path.join(__dir_testing__, "ds_b0", "sub-fieldmap", "fmap", "sub-1_acq-gre_magnitude1.json")
            nii = nib.load(fname_anat)
            with open(fname_json) as f:
                json_data = json.load(f)
            json_data['SliceTiming'] = [1, 0, 2]
            fname_target = os.path.join(tmp, 'target.nii.gz')
            nib.save(nib.Nifti1Image(nii.get_fdata()[:, :, :3], nii.affine, nii.header), fname_target)
            with open(os.path.join(tmp, 'target.json'), 'w') as f:
                json.dump(json_data, f)

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['convert-shim-coefs-format',
                                             '--target', fname_target,
                                             '--input', fname_input,
                                             '--input-file-format', 'slicewise',
                                             '--output-file-format', 'chronological',
                                             '-o', fname_output,
                                             '--reverse-slice-order',
                                             '-v', 'debug'],

                                catch_exceptions=False)
            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == "1.000000, 1.000000, 1.000000, 1.000000,\n"
                assert f.readline() == "0.000000, 0.000000, 0.000000, 0.000000,\n"
                assert f.readline() == "2.000000, 2.000000, 2.000000, 2.000000,\n"

    def test_convert_shim_coefs_sl_cl(self):
        """Test the combine shim coefs function"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            fname_input = os.path.join(tmp, 'shim_coefs.txt')
            with open(fname_input, 'w', encoding='utf-8') as f:
                f.write("12, 14, 16, 18, 15, 11, 12, 13, 14\n")
                f.write("16, 18, 20, 18, 15, 11, 12, 13, 14,\n")

            fname_output = os.path.join(tmp, 'shim_coefs_output.txt')

            runner = CliRunner()
            res = runner.invoke(b0shim_cli, ['convert-shim-coefs-format',
                                             '--input', fname_input,
                                             '--input-file-format', 'slicewise',
                                             '--output-file-format', 'custom-cl',
                                             '-o', fname_output,
                                             '--reverse-slice-order',
                                             '-v', 'debug'],

                                catch_exceptions=False)
            assert res.exit_code == 0
            with open(fname_output, 'r', encoding='utf-8') as f:
                assert f.readline() == "(mA)    xy         zy         zx      x2-y2         z2\n"
                assert f.readline() == "        14         12         11         13         15\n"
                assert f.readline() == "\n"
                assert f.readline() == "(G/cm)     x            y            z      bo (Hz)\n"
                assert f.readline() == "    0.000187     0.000205     0.000185   -16.000000\n"
                assert f.readline() == "    0.000145     0.000164     0.000185   -12.000000\n"
