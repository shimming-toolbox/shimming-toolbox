#!usr/bin/env python3
# -*- coding: utf-8

import json
import logging
import nibabel as nib
import numpy as np
import os
import pytest

from shimmingtoolbox.shim.shim_utils import (dac_to_shim_units, phys_to_shim_cs, shim_to_phys_cs,
                                             calculate_metric_within_mask, logger, phys_to_gradient_cs,
                                             gradient_to_phys_cs)
from shimmingtoolbox.coils.coordinates import get_main_orientation


class TestDacToShimUnits:

    def test_dac_to_shim_units_prisma_fit(self):
        dac_units = {'1': [14436, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Prisma_fit', '167006', dac_units)
        assert np.all(np.isclose(ui_units['1'], [2300, 2300, 2300]))
        assert np.all(np.isclose(ui_units['2'], [4959.01, 3551.29, 3503.299, 3551.29, 3487.302]))

    def test_dac_to_shim_units_investigational_device_7t(self):
        dac_units = {'1': [62479, 62264, 54082], '2': [18000, 18000, 18000, 18000, 18000],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Investigational_Device_7T', '18923', dac_units)
        assert np.all(np.isclose(ui_units['1'], [4999.976, 4999.980, 4999.957]))
        assert np.all(np.isclose(ui_units['2'], [6163.200, 2592.000, 2592.000, 2476.800, 2476.800]))

    def test_dac_to_shim_units_terra(self):
        dac_units = {'1': [17729, 18009, 17872], '2': [12500.0] * 5,
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Terra', '79121', dac_units)
        assert np.all(np.isclose(ui_units['1'], [3000] * 3))
        assert np.all(np.isclose(ui_units['2'], [9360.0, 4680.0, 4620.0, 4620.0, 4560.0]))

    def test_dac_to_shim_units_0(self):
        dac_units = {'1': [0, 0, 0], '2': [0, 0, 0, 0, 0],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Prisma_fit', '167006', dac_units)
        assert np.all(np.isclose(ui_units['1'], [0, 0, 0]))
        assert np.all(np.isclose(ui_units['2'], [0, 0, 0, 0, 0]))

    def test_dac_to_shim_units_unknown_scanner(self, caplog):
        dac_units = {'1': [14436, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}

        with caplog.at_level(logging.DEBUG, logger.name):
            dac_to_shim_units('Unknown', 'Unknown', '167006', dac_units)

        assert "Unknown not implemented or does not include enough metadata information" in caplog.text

    def test_dac_to_shim_units_outside_bounds(self):
        dac_units = {'1': [20000, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}

        with pytest.raises(ValueError, match="Current shim settings exceed known system limits."):
            dac_to_shim_units('Siemens', 'Prisma_fit', '167006', dac_units)


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
    ([1, 2, 3], "TRA", "-", [1, -2, 3]),  # PE: AP
    ([1, 2, 3], "TRA", "", [-1, 2, 3]),  # PE: PA
    ([1, 2, 3], "SAG", "", [3, -2, -1]),  # PE: AP
    ([1, 2, 3], "SAG", "-", [-3, 2, -1]),  # PE: PA
    ([1, 2, 3], "COR", "", [-3, -1, -2]),  # PE: RL
    ([1, 2, 3], "COR", "-", [3, 1, -2])  # PE: LR
])
def test_phys_to_gradient_cs(input_coefs, orientation, phase_encode_dir, expected_coefs, tmpdir):
    fname_target = create_nifti(orientation, phase_encode_dir, tmpdir)
    output_coefs = phys_to_gradient_cs(*input_coefs, fname_target)
    assert np.allclose(output_coefs, expected_coefs)


@pytest.mark.parametrize("expected_coefs, orientation, phase_encode_dir, input_coefs", [
    ([1, 2, 3], "TRA", "-", [1, -2, 3]),  # PE: AP
    ([1, 2, 3], "TRA", "", [-1, 2, 3]),  # PE: PA
    ([1, 2, 3], "SAG", "", [3, -2, -1]),  # PE: AP
    ([1, 2, 3], "SAG", "-", [-3, 2, -1]),  # PE: PA
    ([1, 2, 3], "COR", "", [-3, -1, -2]),  # PE: RL
    ([1, 2, 3], "COR", "-", [3, 1, -2])  # PE: LR
])
def test_gradient_to_phys_cs(input_coefs, orientation, phase_encode_dir, expected_coefs, tmpdir):
    fname_target = create_nifti(orientation, phase_encode_dir, tmpdir)
    output_coefs = gradient_to_phys_cs(*input_coefs, fname_target)
    assert np.allclose(output_coefs, expected_coefs)


def create_nifti(orientation, phase_encode_dir, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    data = np.zeros((5, 5, 5))
    if orientation == "TRA":
        affine = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        dim_info = (0, 1, 2)
        pe_letter = "j"
    elif orientation == "SAG":
        affine = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        dim_info = (1, 0, 2)
        pe_letter = "i"
    elif orientation == "COR":
        affine = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        dim_info = (1, 0, 2)
        pe_letter = "i"
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    nii = nib.Nifti1Image(data, affine)
    nii.header.set_dim_info(*dim_info)
    fname_nii = os.path.join(path_output, f"target_{orientation}_pe{phase_encode_dir}.nii.gz")
    nii.to_filename(fname_nii)
    nii2 = nib.as_closest_canonical(nii)  # Make sure the target is in canonical orientation (RAS)
    nii2.to_filename(fname_nii.replace(".nii.gz", "canon.nii.gz"))

    fname_json = fname_nii.replace(".nii.gz", ".json")
    image_orientation = {
        "TRA": [1, 0, 0, 0, 1, 0],
        "SAG": [0, 1, 0, 0, 0, -1],
        "COR": [1, 0, 0, 0, 0, -1]
    }[orientation]
    assert get_main_orientation(image_orientation) == orientation
    data_json = {
        "Manufacturer": "Siemens",
        "PatientPosition": "HFS",
        "ImageOrientationPatientDICOM": image_orientation,
        "PhaseEncodingDirection": f"{pe_letter}{phase_encode_dir}"
    }
    with open(fname_json, "w") as f:
        json.dump(data_json, f)
    return fname_nii
