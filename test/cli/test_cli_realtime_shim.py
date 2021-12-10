#!usr/bin/env python3
# coding: utf-8
import os
import pathlib
import tempfile
import nibabel as nib
import pytest
import json


from click.testing import CliRunner
from shimmingtoolbox.cli.realtime_shim import realtime_shim_cli
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.cli.realtime_shim import _get_phase_encode_direction_sign

# Path for mag fieldmap
fname_fieldmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
# Path for mag anat image
fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.json')
# Path for resp data
fname_resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime', 'sub-realtime_PMUresp_signal.resp')


def test_cli_realtime_shim():
    """Test CLI for performing realtime shimming experiments"""
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # Path for mag anat image
        nii_anat = nib.load(fname_anat)
        anat = nii_anat.get_fdata()

        # Set up mask
        # Cube
        # static
        nx, ny, nz = anat.shape
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=30, len_dim2=30, len_dim3=nz)

        nii_mask_static = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask_static = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask_static, fname_mask_static)

        # Riro
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=20, len_dim2=20, len_dim3=nz)

        nii_mask_riro = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        fname_mask_riro = os.path.join(tmp, 'mask.nii.gz')
        nib.save(nii_mask_riro, fname_mask_riro)

        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fieldmap,
                                                   '--mask-static', fname_mask_static,
                                                   '--mask-riro', fname_mask_riro,
                                                   '--output', path_output,
                                                   '--resp', fname_resp,
                                                   '--anat', fname_anat],
                               catch_exceptions=False)
        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_cli_realtime_shim_no_mask():
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fieldmap,
                                                   '--output', path_output,
                                                   '--resp', fname_resp,
                                                   '--anat', fname_anat],
                               catch_exceptions=False)

        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_phase_encode_sign():
    # Using this acquisition because it has a positive phase encode direction
    fname_nii = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.nii.gz')
    phase_encode_is_positive = _get_phase_encode_direction_sign(fname_nii)

    assert phase_encode_is_positive is True


def test_phase_encode_wrong_dim():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        nii = nib.load(fname_anat)
        with open(fname_json) as json_file:
            json_data = json.load(json_file)

        json_data['PhaseEncodingDirection'] = "i"
        fname_new_json = os.path.join(tmp, "anat.json")
        with open(fname_new_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        fname_nii = os.path.join(tmp, 'anat.nii.gz')
        nib.save(nii, fname_nii)

        with pytest.raises(RuntimeError,
                           match="Inconsistency between dim_info of fieldmap and PhaseEncodeDirection tag in the json"):
            _get_phase_encode_direction_sign(fname_nii)


def test_phase_encode_wrong_tag_value():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:

        nii = nib.load(fname_anat)
        with open(fname_json) as json_file:
            json_data = json.load(json_file)

        json_data['PhaseEncodingDirection'] = "j----"
        fname_new_json = os.path.join(tmp, "anat.json")
        with open(fname_new_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        fname_nii = os.path.join(tmp, 'anat.nii.gz')
        nib.save(nii, fname_nii)

        with pytest.raises(ValueError,
                           match="Unexpected value for PhaseEncodingDirection:"):
            _get_phase_encode_direction_sign(fname_nii)


def test_cli_realtime_shim_sag_anat():
    """We do not have a sagittal orientation in testing_data so we change the json manually to test for the SAG case"""
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Change json to have an orientation that is SAGITTAL
        nii = nib.load(fname_anat)
        with open(fname_json) as json_file:
            json_data = json.load(json_file)
        json_data['ImageOrientationPatientDICOM'] = [0, 1, 0, 0, 0, -1]
        fname_json_sag = os.path.join(tmp, 'anat_sag.json')
        with open(fname_json_sag, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        fname_anat_sag = os.path.join(tmp, 'anat_sag.nii.gz')
        nib.save(nii, fname_anat_sag)

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fieldmap,
                                                   '--output', path_output,
                                                   '--resp', fname_resp,
                                                   '--anat', fname_anat_sag],
                               catch_exceptions=False)

        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_cli_realtime_shim_cor_anat():
    """We do not have a coronal orientation in testing_data so we change the json manually to test for the COR case"""
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Change json to have an orientation that is CORONAL
        nii = nib.load(fname_anat)
        with open(fname_json) as json_file:
            json_data = json.load(json_file)
        json_data['ImageOrientationPatientDICOM'] = [1, 0, 0, 0, 0, -1]
        fname_json_cor = os.path.join(tmp, 'anat_cor.json')
        with open(fname_json_cor, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        fname_anat_cor = os.path.join(tmp, 'anat_cor.nii.gz')
        nib.save(nii, fname_anat_cor)

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fieldmap,
                                                   '--output', path_output,
                                                   '--resp', fname_resp,
                                                   '--anat', fname_anat_cor],
                               catch_exceptions=False)

        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0


def test_cli_realtime_shim_tra_orient_text():
    """Add a json tag ImageOrientationText with 'Tra'"""
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # Specify output for text file and figures
        path_output = os.path.join(tmp, 'test_realtime_shim')

        # Add tag to json that says it is axial
        nii = nib.load(fname_anat)
        with open(fname_json) as json_file:
            json_data = json.load(json_file)
        json_data['ImageOrientationText'] = 'Tra'
        fname_json_text = os.path.join(tmp, 'anat_text.json')
        with open(fname_json_text, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        fname_anat_text = os.path.join(tmp, 'anat_text.nii.gz')
        nib.save(nii, fname_anat_text)

        # Run the CLI
        result = runner.invoke(realtime_shim_cli, ['--fmap', fname_fieldmap,
                                                   '--output', path_output,
                                                   '--resp', fname_resp,
                                                   '--anat', fname_anat_text],
                               catch_exceptions=False)

        assert len(os.listdir(path_output)) != 0
        assert result.exit_code == 0
