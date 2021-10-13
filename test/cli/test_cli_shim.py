#!/usr/bin/python3
# -*- coding: utf-8 -*
import pytest
from click.testing import CliRunner
import tempfile
import pathlib
import os
import nibabel as nib
import numpy as np
from shutil import copy
import json

from shimmingtoolbox.cli.shim import define_slices_cli
from shimmingtoolbox.cli.shim import shim_cli
from shimmingtoolbox.cli.realtime_shim import realtime_shim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox import __dir_config_scanner_constraints__
from shimmingtoolbox.simulate.numerical_model import NumericalModel


def _define_inputs(fmap_dim):
    # fname for fmap
    fname_fmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_fieldmap.nii.gz')
    nii = nib.load(fname_fmap)

    fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                              'sub-example_fieldmap.json')
    data = json.load(open(fname_json))

    if fmap_dim == 4:
        nii_fmap = nii
    elif fmap_dim == 3:
        nii_fmap = nib.Nifti1Image(np.mean(nii.get_fdata(), axis=3), nii.affine, header=nii.header)
    elif fmap_dim == 2:
        nii_fmap = nib.Nifti1Image(nii.get_fdata()[..., 0, 0], nii.affine, header=nii.header)
    else:
        raise ValueError("Supported Dimensions are 2, 3 or 4")

    # fname for anat
    fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                              'sub-example_unshimmed_e1.nii.gz')
    nii_anat = nib.load(fname_anat)
    anat = nii_anat.get_fdata()

    # Set up mask: Cube
    # static
    nx, ny, nz = anat.shape
    mask = shapes(anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)

    nii_mask = nib.Nifti1Image(mask.astype(int), nii_anat.affine)

    return nii_fmap, nii_anat, nii_mask, data


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask,data", [(
            _define_inputs(fmap_dim=3)
    )]
)
class TestCliStatic(object):
    def test_cli_static_sph(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))

    def test_cli_static_no_mask(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))

    def test_cli_static_coils(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with input coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            nii_dummy_coil = nib.Nifti1Image(np.repeat(nii_fmap.get_fdata()[..., np.newaxis], 9, axis=3),
                                             nii_fmap.affine, header=nii_fmap.header)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            runner = CliRunner()
            # TODO: use actual coil files (These are just dummy files to test if the code works)
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--coil', fname_dummy_coil, __dir_config_scanner_constraints__,
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))

    def test_cli_static_coils_and_sph(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with input coil and scanner coil"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            nii_dummy_coil = nib.Nifti1Image(np.repeat(nii_fmap.get_fdata()[..., np.newaxis], 9, axis=3),
                                             nii_fmap.affine, header=nii_fmap.header)
            fname_dummy_coil = os.path.join(tmp, 'dummy_coil.nii.gz')
            nib.save(nii_dummy_coil, fname_dummy_coil)

            runner = CliRunner()
            # TODO: use actual coil files (These are just dummy files to test if the code works)
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--coil', fname_dummy_coil, __dir_config_scanner_constraints__,
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil1_siemens_gradient_coil.txt"))

    def test_cli_static_format_chronological_coil(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil with chronological-coil oformat"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--slice-factor', '2',
                                           '--output-file-format-scanner', 'chronological-coil',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))
            # There should be 10 x 4 values

    def test_cli_static_format_chronological_ch(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil with hronological_ch o_format"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--slice-factor', '2',
                                           '--output-file-format-scanner', 'chronological-ch',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))
            # There should be 4 x 10 x 1 value

    def test_cli_static_format_slicewise_ch(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil with slicewise_ch oformat"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '--slice-factor', '2',
                                           '--output-file-format-scanner', 'slicewise-ch',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))

    def test_cli_static_debug_verbose(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_static',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask', fname_mask,
                                           '--scanner-coil-order', '1',
                                           '-v', 'debug',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))
            # Artefacts of the debug verbose
            assert os.path.isfile(os.path.join(tmp, "tmp_extended_fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "anat.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "mask.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fig_currents.png"))

    def test_cli_static_no_coil(self, nii_fmap, nii_anat, nii_mask, data):
        """Test cli with scanner coil profiles of order 1 with default constraints"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the modified fieldmap (one volume)
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            runner = CliRunner()

            with pytest.raises(RuntimeError, match="No custom or scanner coils were selected."):
                runner.invoke(shim_cli, ['fieldmap_static',
                                         '--fmap', fname_fmap,
                                         '--anat', fname_anat,
                                         '--mask', fname_mask,
                                         '--output', tmp],
                              catch_exceptions=False)

    # def test_cli_static_order_0(self, nii_fmap, nii_anat, nii_mask, data):
    #     """Test cli with scanner coil profiles of order 1 with default constraints"""
    #     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
    #         # Save the modified fieldmap (one volume)
    #         fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
    #         nib.save(nii_fmap, fname_fmap)
    #         # Save json
    #         fname_json = os.path.join(tmp, 'fmap.json')
    #         with open(fname_json, 'w', encoding='utf-8') as f:
    #             json.dump(data, f, indent=4)
    #         # Save the mask
    #         fname_mask = os.path.join(tmp, 'mask.nii.gz')
    #         nib.save(nii_mask, fname_mask)
    #         # Save the anat
    #         fname_anat = os.path.join(tmp, 'anat.nii.gz')
    #         nib.save(nii_anat, fname_anat)
    #
    #         runner = CliRunner()
    #         runner.invoke(shim_cli, ['fieldmap_static',
    #                                  '--fmap', fname_fmap,
    #                                  '--anat', fname_anat,
    #                                  '--mask', fname_mask,
    #                                  '--scanner-coil-order', '0',
    #                                  '--output-value-format', 'absolute',
    #                                  '--output', tmp],
    #                       catch_exceptions=False)
    #
    #         assert os.path.isfile(os.path.join(tmp, "coefs_coil0_siemens_gradient_coil.txt"))


@pytest.mark.parametrize(
    "nii_fmap,nii_anat,nii_mask,data", [(
            _define_inputs(fmap_dim=4)
    )]
)
class TestCLIRealtime(object):
    def test_cli_rt_sph(self, nii_fmap, nii_anat, nii_mask, data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the fieldmap
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

            # Copy fieldmap json to tmp
            fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                           'sub-example_fieldmap.json')
            copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_realtime',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask-static', fname_mask,
                                           '--mask-riro', fname_mask,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))

    def test_cli_rt_debug(self, nii_fmap, nii_anat, nii_mask, data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the fieldmap
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

            # Copy fieldmap json to tmp
            fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                           'sub-example_fieldmap.json')
            copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_realtime',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask-static', fname_mask,
                                           '--mask-riro', fname_mask,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '-v', 'debug',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))
            # Artefacts of the debug verbose
            assert os.path.isfile(os.path.join(tmp, "tmp_extended_fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fmap.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "anat.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "mask.nii.gz"))
            assert os.path.isfile(os.path.join(tmp, "fig_currents.png"))

    def test_cli_rt_no_mask(self, nii_fmap, nii_anat, nii_mask, data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the fieldmap
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

            # Copy fieldmap json to tmp
            fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                           'sub-example_fieldmap.json')
            copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_realtime',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))

    def test_cli_rt_chronological_ch(self, nii_fmap, nii_anat, nii_mask, data):
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Save the fieldmap
            fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
            nib.save(nii_fmap, fname_fmap)
            # Save json
            fname_json = os.path.join(tmp, 'fmap.json')
            with open(fname_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Save the mask
            fname_mask = os.path.join(tmp, 'mask.nii.gz')
            nib.save(nii_mask, fname_mask)
            # Save the anat
            fname_anat = os.path.join(tmp, 'anat.nii.gz')
            nib.save(nii_anat, fname_anat)

            # Input pmu fname
            fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')

            # Copy fieldmap json to tmp
            fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                           'sub-example_fieldmap.json')
            copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))

            runner = CliRunner()
            res = runner.invoke(shim_cli, ['fieldmap_realtime',
                                           '--fmap', fname_fmap,
                                           '--anat', fname_anat,
                                           '--mask-static', fname_mask,
                                           '--mask-riro', fname_mask,
                                           '--resp', fname_resp,
                                           '--slice-factor', '2',
                                           '--scanner-coil-order', '1',
                                           '--output-file-format', 'chronological-ch',
                                           '--output', tmp],
                                catch_exceptions=False)

            assert res.exit_code == 0
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch0_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch1_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch2_siemens_gradient_coil.txt"))
            assert os.path.isfile(os.path.join(tmp, "coefs_coil0_ch3_siemens_gradient_coil.txt"))


def test_cli_define_slices_def():
    """Test using a number for the number of slices"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_output = os.path.join(tmp, 'slices.json')
        res = runner.invoke(define_slices_cli, ['--slices', '12',
                                                '--factor', '5',
                                                '--method', 'sequential',
                                                '-o', fname_output],
                            catch_exceptions=False)

        assert res.exit_code == 0
        assert os.path.isfile(fname_output)


def test_cli_define_slices_anat():
    """Test using an anatomical file"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()
        fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                                  'sub-example_unshimmed_e1.nii.gz')
        fname_output = os.path.join(tmp, 'slices.json')
        res = runner.invoke(define_slices_cli, ['--slices', fname_anat,
                                                '--factor', '5',
                                                '--method', 'sequential',
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
                                              '--method', 'sequential',
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
                                              '--method', 'sequential',
                                              '-o', fname_output],
                          catch_exceptions=False)


# def test_grad_realtime_shim_vs_fieldmap_realtime_shim():
#     """Test to compare grad vs fieldmap realtime shim"""
#     nii_fmap, nii_anat, nii_mask, data = _define_inputs(4)
#
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         # Save the fieldmap
#         fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
#         nib.save(nii_fmap, fname_fmap)
#         # Save json
#         fname_json = os.path.join(tmp, 'fmap.json')
#         with open(fname_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         # Save the mask
#         fname_mask = os.path.join(tmp, 'mask.nii.gz')
#         nib.save(nii_mask, fname_mask)
#         # Save the anat
#         fname_anat = os.path.join(tmp, 'anat.nii.gz')
#         nib.save(nii_anat, fname_anat)
#
#         # Input pmu fname
#         fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
#
#         # Copy fieldmap json to tmp
#         fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
#                                        'sub-example_fieldmap.json')
#         copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))
#
#         runner = CliRunner()
#
#         # fieldmap rt shim
#         runner.invoke(shim_cli, ['fieldmap_realtime',
#                                  '--fmap', fname_fmap,
#                                  '--anat', fname_anat,
#                                  '--mask-static', fname_mask,
#                                  '--mask-riro', fname_mask,
#                                  '--resp', fname_resp,
#                                  '--slice-factor', '1',
#                                  '--mask-dilation-kernel-size', '3',
#                                  '--optimizer-method', 'least_squares',
#                                  '--scanner-coil-order', '1',
#                                  '--output', os.path.join(tmp, 'fmap'),
#                                  '-v', 'debug'],
#                       catch_exceptions=False)
#
#         # grad rt shim
#         result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fmap,
#                                                    '--anat', fname_anat,
#                                                    '--mask-static', fname_mask,
#                                                    '--mask-riro', fname_mask,
#                                                    '--output', os.path.join(tmp, 'grad'),
#                                                    '--resp', fname_resp],
#                                catch_exceptions=False)
#         a=1
#
#
# def test_grad_vs_fieldmap_known_result():
#
#     nii_fmap, nii_anat, nii_mask, data = _define_inputs(4)
#
#     def create_fieldmap(n_slices=3):
#         # Set up 2-dimensional unshimmed fieldmaps
#         num_vox = 100
#         model_obj = NumericalModel('shepp-logan', num_vox=num_vox)
#         model_obj.generate_deltaB0('linear', [1, 0])
#         tr = 0.025  # in s
#         te = [0.004, 0.008]  # in s
#         model_obj.simulate_measurement(tr, te)
#         phase_meas1 = model_obj.get_phase()
#         phase_e1 = phase_meas1[:, :, 0, 0]
#         phase_e2 = phase_meas1[:, :, 0, 1]
#         b0_map = ((phase_e2 - phase_e1) / (te[1] - te[0])) / (2 * np.pi)
#
#         # Construct a 3-dimensional synthetic field map by stacking different z-slices along the 3rd dimension. Each
#         # slice is subjected to a manipulation of model_obj across slices (e.g. rotation, squared) in order to test
#         # various shim configurations.
#         unshimmed = np.zeros([num_vox, num_vox, n_slices])
#         for i_n in range(n_slices):
#             unshimmed[:, :, i_n] = b0_map
#
#         return unshimmed
#
#     fmap = np.repeat(create_fieldmap(3)[..., np.newaxis], 10, 3)
#     nii_fmap = nib.Nifti1Image(fmap, nii_fmap.affine, header=nii_fmap.header)
#
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         # Save the fieldmap
#         fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
#         nib.save(nii_fmap, fname_fmap)
#         # Save json
#         fname_json = os.path.join(tmp, 'fmap.json')
#         with open(fname_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         # Save the mask
#         fname_mask = os.path.join(tmp, 'mask.nii.gz')
#         nib.save(nii_mask, fname_mask)
#         # Save the anat
#         fname_anat = os.path.join(tmp, 'anat.nii.gz')
#         nib.save(nii_anat, fname_anat)
#
#         # Input pmu fname
#         fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
#
#         # Copy fieldmap json to tmp
#         fname_fmap_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
#                                        'sub-example_fieldmap.json')
#         copy(fname_fmap_json, os.path.join(tmp, 'fmap.json'))
#
#         runner = CliRunner()
#
#         # fieldmap rt shim
#         runner.invoke(shim_cli, ['fieldmap_realtime',
#                                  '--fmap', fname_fmap,
#                                  '--anat', fname_anat,
#                                  '--mask-static', fname_mask,
#                                  '--mask-riro', fname_mask,
#                                  '--resp', fname_resp,
#                                  '--slice-factor', '1',
#                                  '--mask-dilation-kernel-size', '3',
#                                  '--optimizer-method', 'least_squares',
#                                  '--scanner-coil-order', '1',
#                                  '--output', os.path.join(tmp, 'fmap'),
#                                  '-v', 'debug'],
#                       catch_exceptions=False)
#
#         # grad rt shim
#         result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fmap,
#                                                    '--anat', fname_anat,
#                                                    '--mask-static', fname_mask,
#                                                    '--mask-riro', fname_mask,
#                                                    '--output', os.path.join(tmp, 'grad'),
#                                                    '--resp', fname_resp],
#                                catch_exceptions=False)
#         a=1


# def test_static_shim_known_real_input():
#     """Test to validate using acdc82
#
#     Currently validated the output for order 1 using an anat in the same space as fmap, anat in different space as fmap
#     (raised mask problems if fmap has bigger voxels than anat)
#     """
#     fname_fmap = "/Users/alex/Documents/School/Polytechnique/Master/project/Data/acdc_82p_nifti/Gz+50/derivatives/fieldmap-Gz+50.nii.gz"
#     nii_fmap = nib.load(fname_fmap)
#     fname_json = "/Users/alex/Documents/School/Polytechnique/Master/project/Data/acdc_82p_nifti/Gz+50/derivatives/fieldmap-Gz+50.json"
#     data = json.load(open(fname_json))
#
#     # TODO: validate using anat in difference space than fmap
#     anat = np.ones_like(nii_fmap.get_fdata())
#     anat[0, 0, 0] = 0
#     anat_affine = [[-1, 0, 0, 100],
#                    [0, 1, 0, -130],
#                    [0, 0, 2.25, -177],
#                    [0, 0, 0, 1]]
#     nii_anat = nib.Nifti1Image(anat, anat_affine, header=nii_fmap.header)
#
#     nx, ny, nz = anat.shape
#     mask = shapes(anat, 'cube',
#                   center_dim1=int(nx / 2),
#                   center_dim2=int(ny / 2),
#                   len_dim1=20, len_dim2=40, len_dim3=nz)
#
#     nii_mask = nib.Nifti1Image(mask, nii_anat.affine, header=nii_anat.header)
#
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         # Save the fieldmap
#         fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
#         nib.save(nii_fmap, fname_fmap)
#         # Save json
#         fname_json = os.path.join(tmp, 'fmap.json')
#         with open(fname_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         # Save the mask
#         fname_mask = os.path.join(tmp, 'mask.nii.gz')
#         nib.save(nii_mask, fname_mask)
#         # Save the anat
#         fname_anat = os.path.join(tmp, 'anat.nii.gz')
#         nib.save(nii_anat, fname_anat)
#
#         runner = CliRunner()
#         runner.invoke(shim_cli, ['fieldmap_static',
#                                  '--fmap', fname_fmap,
#                                  '--anat', fname_anat,
#                                  '--mask', fname_mask,
#                                  '--scanner-coil-order', '1',
#                                  '--slice-factor', '1',
#                                  '--optimizer-method', 'least_squares',
#                                  '--output', tmp,
#                                  '-v', 'debug'],
#                       catch_exceptions=False)
#
#         a=1
#
#
# def test_rt_shim_known_real_input():
#     """Test to validate using acdc82
#
#     Currently validated the output for order 1 using an anat in the same space as fmap, anat in different space as fmap
#     (raised mask problems if fmap has bigger voxels than anat)
#     """
#     fname_fmap = "/Users/alex/Documents/School/Polytechnique/Master/project/Data/acdc_82p_nifti/Gz+50/derivatives/fieldmap-Gz+50.nii.gz"
#     nii = nib.load(fname_fmap)
#     fmap = np.repeat(nii.get_fdata()[..., np.newaxis], 4, 3)
#     nii_fmap = nib.Nifti1Image(fmap, nii.affine, header=nii.header)
#     fname_json = "/Users/alex/Documents/School/Polytechnique/Master/project/Data/acdc_82p_nifti/Gz+50/derivatives/fieldmap-Gz+50.json"
#     data = json.load(open(fname_json))
#
#     anat = np.ones(nii_fmap.shape[:3])
#     anat[0, 0, 0] = 0
#     anat_affine = [[0, -1, 0, 60],
#                    [0, 0, -2, 20],
#                    [-4, 0, 0, 240],
#                    [0, 0, 0, 1]]
#     nii_anat = nib.Nifti1Image(anat, anat_affine, header=nii_fmap.header)
#
#     nx, ny, nz = anat.shape
#     mask = shapes(anat, 'cube',
#                   center_dim1=int(nx / 2),
#                   center_dim2=int(ny / 2),
#                   len_dim1=20, len_dim2=40, len_dim3=int(nz/2))
#
#     nii_mask = nib.Nifti1Image(mask, nii_anat.affine, header=nii_anat.header)
#
#     with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
#         # Save the fieldmap
#         fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
#         nib.save(nii_fmap, fname_fmap)
#         # Save json
#         fname_json = os.path.join(tmp, 'fmap.json')
#         with open(fname_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         # Save the mask
#         fname_mask = os.path.join(tmp, 'mask.nii.gz')
#         nib.save(nii_mask, fname_mask)
#         # Save the anat
#         fname_anat = os.path.join(tmp, 'anat.nii.gz')
#         nib.save(nii_anat, fname_anat)
#
#         # Input pmu fname
#         fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
#
#         runner = CliRunner()
#         runner.invoke(shim_cli, ['fieldmap_realtime',
#                                  '--fmap', fname_fmap,
#                                  '--anat', fname_anat,
#                                  '--mask-static', fname_mask,
#                                  '--mask-riro', fname_mask,
#                                  '--resp', fname_resp,
#                                  '--slice-factor', '10',
#                                  '--mask-dilation-kernel-size', '3',
#                                  '--optimizer-method', 'least_squares',
#                                  '--scanner-coil-order', '1',
#                                  '--output-file-format', 'chronological-ch',
#                                  '--output', os.path.join(tmp, 'fmap'),
#                                  '-v', 'debug'],
#                       catch_exceptions=False)
#
#         # result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fmap,
#         #                                            '--anat', fname_anat,
#         #                                            '--mask-static', fname_mask,
#         #                                            '--mask-riro', fname_mask,
#         #                                            '--output', os.path.join(tmp, 'grad'),
#         #                                            '--resp', fname_resp],
#         #                        catch_exceptions=False)
#         a=1
