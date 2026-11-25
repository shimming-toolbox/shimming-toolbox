#!usr/bin/env python3
# -*- coding: utf-8

import json
import nibabel as nib
import numpy as np
import os
import pytest

from shimmingtoolbox.shim.shim_utils import (phys_to_shim_cs, shim_to_phys_cs,
                                             calculate_metric_within_mask, phys_to_gradient_cs,
                                             gradient_to_phys_cs, get_flip_matrix)
from shimmingtoolbox.coils.coordinates import get_main_orientation


def test_phys_to_shim_cs():
    out = phys_to_shim_cs(np.array([1, 1, 1]), 'Siemens', orders=(1,))
    assert np.all(out == [-1, 1, -1])


def test_shim_to_phys_cs():
    out = shim_to_phys_cs(np.array([1, 1, 1]), 'Siemens', orders=(1,))
    assert np.all(out == [-1, 1, -1])


class TestCalculateMetricWithinMask:

    def test_calculate_metric_within_mask_mean(self):
        """Test the 'mean' metric calculation with a weighted mask"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        result = calculate_metric_within_mask(array, mask, metric='mean')
        expected_result = (1*1 + 2*0.5 + 3*1 + 4*0.75) / (1 + 0.5 + 1 + 0.75)
        assert np.isclose(result, expected_result)

    def test_calculate_metric_within_mask_std(self):
        """Test the 'std' (standard deviation) metric calculation with a weighted mask"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        result = calculate_metric_within_mask(array, mask, metric='std')
        mean_weighted = np.average(array, weights=mask)
        variance_weighted = np.average((array - mean_weighted) ** 2, weights=mask)
        expected_result = np.sqrt(variance_weighted)
        assert np.isclose(result, expected_result)

    def test_calculate_metric_within_mask_mae(self):
        """Test the 'mae' (mean absolute error) metric calculation with a weighted mask"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        result = calculate_metric_within_mask(array, mask, metric='mae')
        expected_result = np.average(np.abs(array), weights=mask)
        assert np.isclose(result, expected_result)

    def test_calculate_metric_within_mask_mse(self):
        """Test the 'mse' (mean squared error) metric calculation with a weighted mask"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        result = calculate_metric_within_mask(array, mask, metric='mse')
        expected_result = np.average(np.square(array), weights=mask)
        assert np.isclose(result, expected_result)

    def test_calculate_metric_within_mask_rmse(self):
        """Test the 'rmse' (root mean squared error) metric calculation with a weighted mask"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        result = calculate_metric_within_mask(array, mask, metric='rmse')
        mse_weighted = np.average(np.square(array), weights=mask)
        expected_result = np.sqrt(mse_weighted)
        assert np.isclose(result, expected_result)

    def test_calculate_metric_within_mask_invalid_metric(self):
        """Test with an invalid metric that should raise an exception"""
        array = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 0.5, 1, 0.75, 0])

        with pytest.raises(NotImplementedError, match="Metric 'invalid' not implemented. Available metrics:"):
            calculate_metric_within_mask(array, mask, metric='invalid')



@pytest.mark.parametrize("input_coefs, orientation, phase_encode_dir, expected_coefs", [
    ([1, 2, 3], "TRA", "j-", [1, -2, 3]),  # PE: AP
    ([1, 2, 3], "TRA", "j", [-1, 2, 3]),  # PE: PA
    ([1, 2, 3], "SAG", "i", [3, -2, -1]),  # PE: AP
    ([1, 2, 3], "SAG", "i-", [-3, 2, -1]),  # PE: PA
    ([1, 2, 3], "COR", "i", [-3, -1, -2]),  # PE: RL
    ([1, 2, 3], "COR", "i-", [3, 1, -2])  # PE: LR
])
def test_phys_to_gradient_cs(input_coefs, orientation, phase_encode_dir, expected_coefs, tmpdir):
    fname_target = create_nifti(orientation, phase_encode_dir, tmpdir)
    output_coefs = phys_to_gradient_cs(*input_coefs, fname_target)
    assert np.allclose(output_coefs, expected_coefs)


@pytest.mark.parametrize("input_coefs, orientation, phase_encode_dir, expected_coefs", [
    ([1, 10, 100], "TRA", "j-", [-1, -10, -100]),  # PE: AP
    ([1, 10, 100], "TRA", "j", [1, 10, -100]),  # PE: PA
    ([1, 10, 100], "TRA", "i-", [10, -1, -100]),  # PE: LR
    ([1, 10, 100], "TRA", "i", [-10, 1, -100]),  # PE: RL
    ([1, 10, 100], "SAG", "i", [-100, -10, 1]),  # PE: AP
    ([1, 10, 100], "SAG", "i-", [100, 10, 1]),  # PE: PA
    ([1, 10, 100], "SAG", "j-", [-10, 100, 1]),  # PE: HF
    ([1, 10, 100], "SAG", "j", [10, -100, 1]),  # PE: FH
    ([1, 10, 100], "COR", "i", [100, 1, -10]),  # PE: RL
    ([1, 10, 100], "COR", "i-", [-100, -1, -10]),  # PE: LR
    ([1, 10, 100], "COR", "j-", [-1, -100, -10]),  # PE: HF
    ([1, 10, 100], "COR", "j", [1, -100, -10]),  # PE: FH
    ([1, 10, 100], "TRA>SAG20", "j-", [33.2623, -10.0, -94.3113]),  # PE: AP
    ([1, 10, 100], "TRA>COR20", "j-", [-1.0, -43.5989, -90.5491]),  # PE: AP
])
def test_phys_lai_to_gradient_cs(input_coefs, orientation, phase_encode_dir, expected_coefs, tmpdir):
    # This test is somewhat redundant with test_phys_to_gradient_cs but can be more easily confirmed on IDEA
    fname_target = create_nifti(orientation, phase_encode_dir, tmpdir)
    coefs_ras = shim_to_phys_cs(np.array(input_coefs), "Siemens", [1, ])
    output_coefs = phys_to_gradient_cs(*coefs_ras, fname_target)
    assert np.allclose(output_coefs, expected_coefs)


@pytest.mark.parametrize("expected_coefs, orientation, phase_encode_dir, input_coefs", [
    ([1, 2, 3], "TRA", "j-", [1, -2, 3]),  # PE: AP
    ([1, 2, 3], "TRA", "j", [-1, 2, 3]),  # PE: PA
    ([1, 2, 3], "SAG", "i", [3, -2, -1]),  # PE: AP
    ([1, 2, 3], "SAG", "i-", [-3, 2, -1]),  # PE: PA
    ([1, 2, 3], "COR", "i", [-3, -1, -2]),  # PE: RL
    ([1, 2, 3], "COR", "i-", [3, 1, -2])  # PE: LR
])
def test_gradient_to_phys_cs(input_coefs, orientation, phase_encode_dir, expected_coefs, tmpdir):
    fname_target = create_nifti(orientation, phase_encode_dir, tmpdir)
    output_coefs = gradient_to_phys_cs(*input_coefs, fname_target)
    assert np.allclose(output_coefs, expected_coefs)


def create_nifti(orientation, phase_encode_dir, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if phase_encode_dir[0] == "i":
        dim_info = (1, 0, 2)
    elif phase_encode_dir[0] == "j":
        dim_info = (0, 1, 2)
    else:
        raise ValueError(f"Unknown phase encoding direction: {phase_encode_dir}")

    data = np.zeros((5, 5, 5))
    if orientation == "TRA":
        affine = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif orientation == "SAG":
        affine = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    elif orientation == "COR":
        affine = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    elif orientation == "TRA>SAG20":
        affine = np.array(
            [[-2.05557775, 0.    ,  1.02605963,  124.88800049],
             [-0.        , 2.1875, -0.        , -108.96459961],
             [ 0.74816859, 0.    ,  2.81907773, -68.61650085 ],
             [ 0.        , 0.    ,  0.        ,  1.          ]]
        )
    elif orientation == "TRA>COR20":
        affine = np.array([[ -2.1875,  0.        , -0.        ,  140.        ],
                           [  0.    ,  2.05557775,  1.02602792, -110.22515869],
                           [  0.    , -0.74816859,  2.81899047,  9.94129181  ],
                           [  0.    ,  0.        ,  0.        ,  1.          ]])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    nii = nib.Nifti1Image(data, affine)
    nii.header.set_dim_info(*dim_info)
    fname_nii = os.path.join(path_output, f"target_{orientation.replace('>', '-')}_pe{phase_encode_dir}.nii.gz")
    nii.to_filename(fname_nii)
    nii2 = nib.as_closest_canonical(nii)  # Make sure the target is in canonical orientation (RAS)
    nii2.to_filename(fname_nii.replace(".nii.gz", "canon.nii.gz"))

    fname_json = fname_nii.replace(".nii.gz", ".json")
    image_orientation = {
        "TRA": [1, 0, 0, 0, 1, 0],
        "SAG": [0, 1, 0, 0, 0, -1],
        "COR": [1, 0, 0, 0, 0, -1],
        "TRA>SAG20": [0.939693, 0, 0.34202, 0, 1, 0],
        "TRA>COR20": [1, 0, 0, 0, 0.939693, 0.34202]
    }[orientation]
    assert get_main_orientation(image_orientation) == orientation[:3]
    data_json = {
        "Manufacturer": "Siemens",
        "PatientPosition": "HFS",
        "ImageOrientationPatientDICOM": image_orientation,
        "PhaseEncodingDirection": f"{phase_encode_dir}"
    }
    with open(fname_json, "w") as f:
        json.dump(data_json, f)
    return fname_nii


class TestGetFlipMatrix:
    def test_flip_cs(self):
        out = get_flip_matrix('RAS', orders=[1, ])
        assert np.all(out == [1, 1, 1])

    def test_flip_cs_lpi(self):
        out = get_flip_matrix('LPI', orders=[1, ])
        assert np.all(out == [-1, -1, -1])

    def test_flip_cs_order2(self):
        out = get_flip_matrix('LAI', orders=[1, 2])
        assert np.all(out == [1, -1, -1, -1, -1, 1, 1, 1])

    def test_flip_cs_len4(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAIS')

    def test_flip_cs_lap(self):
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            get_flip_matrix('LAP')

    def test_flip_siemens(self):
        out = get_flip_matrix('LAI', orders=[1, 2, 3], manufacturer='Siemens')
        # TODO: Verify 3rd order
        assert np.all(out == [-1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1])

    def test_flip_ge(self):
        out = get_flip_matrix('LPI', orders=[1, 2], manufacturer='GE')
        assert np.all(out == [-1, -1, -1, 1, 1, 1, 1, 1])

    def test_flip_philips(self):
        out = get_flip_matrix('RPI', orders=[1, 2], manufacturer='PHILIPS')
        assert np.all(out == [1, -1, -1, 1, -1, 1, 1, -1])
