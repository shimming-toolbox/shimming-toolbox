#!/usr/bin/env python3

from click.testing import CliRunner
from shimmingtoolbox.cli.mask_shimmingtoolbox import cube, square, mask_threshold


def test_cli_mask_cube():
    runner = CliRunner()
    dim1 = 50
    dim2 = 30
    dim3 = 5
    result = runner.invoke(cube, ['-fname_data', 'C:/Users/heuss/sub-example_unshimmed_e1.nii.gz', '-len_dim1',
                                  dim1, '-len_dim2', dim2, '-len_dim3', dim3])
    assert result.exit_code == 0
    assert result is not None


def test_cli_mask_square():
    runner = CliRunner()
    dim1 = 50
    dim2 = 30
    result = runner.invoke(square, ['-fname_data', 'C:/Users/heuss/sub-example_unshimmed_e1.nii.gz', '-len_dim1',
                                    dim1, '-len_dim2', dim2])
    assert result.exit_code == 0
    assert result is not None


def test_cli_mak_threshold():
    runner = CliRunner()
    thr = 30
    result = runner.invoke(mask_threshold, ['-fname_data', 'C:/Users/heuss/sub-example_unshimmed_e1.nii.gz', '-thr',
                                            thr])
    assert result.exit_code == 0
    assert result is not None
