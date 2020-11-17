#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import os
import pathlib
import tempfile

from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli
from shimmingtoolbox import __dir_testing__


# Add pytest prelude dependency
def test_cli_prepare_fieldmap_1_echo():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_phasediff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                       'sub-example_phasediff.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '-output', fname_output],
                               catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)

        # Debug
        import nibabel as nib
        fieldmap_prelude = nib.load(fname_output).get_fdata()
        fname_fsl = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_fieldmap.nii.gz')
        fieldmap_fsl = nib.load(fname_fsl).get_fdata()
        a=1


# Add pytest prelude dependency
def test_cli_prepare_fieldmap_2_echos():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        fname_phase1 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        fname_phase2 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        fname_output = os.path.join(tmp, 'fieldmap.nii.gz')

        fname_mag = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phase1, fname_phase2, '-mag', fname_mag,
                                                      '-output', fname_output], catch_exceptions=False)

        assert result.exit_code == 0
        assert os.path.isfile(fname_output)

        # Debug
        import nibabel as nib
        fieldmap_prelude = nib.load(fname_output).get_fdata()
        fname_fsl = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_fieldmap.nii.gz')
        fieldmap_fsl = nib.load(fname_fsl).get_fdata()
        a=1