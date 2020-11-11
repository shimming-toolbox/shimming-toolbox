#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import os
import pathlib
import tempfile

from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli
from shimmingtoolbox import __dir_testing__


def test_cli_prepare_fieldmap():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        runner = CliRunner()

        # TODO: use phase 1 and phase 2 from sub_fieldmap
        fname_phasediff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                       'sub-example_phasediff.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        # fname_phase1 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        # fname_phase2 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        # fname_mag = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')

        result = runner.invoke(prepare_fieldmap_cli, [fname_phasediff, '-mag', fname_mag, '-output', tmp],
                               catch_exceptions=False)

    assert result.exit_code == 0
