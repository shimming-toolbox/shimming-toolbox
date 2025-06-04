#!usr/bin/env python3
# -*- coding: utf-8

import logging
import numpy as np
import pytest

from shimmingtoolbox.shim.shim_utils import dac_to_shim_units, phys_to_shim_cs, shim_to_phys_cs, calculate_metric_within_mask, logger


class TestDacToShimUnits:

    def test_dac_to_shim_units_prisma_fit(self):
        dac_units = {'1': [14436, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Prisma_fit', dac_units)
        assert np.all(np.isclose(ui_units['1'], [2300, 2300, 2300]))
        assert np.all(np.isclose(ui_units['2'], [4959.01, 3551.29, 3503.299, 3551.29, 3487.302]))

    def test_dac_to_shim_units_investigational_device_7t(self):
        dac_units = {'1': [62479, 62264, 54082], '2': [18000, 18000, 18000, 18000, 18000],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Investigational_Device_7T', dac_units)
        assert np.all(np.isclose(ui_units['1'], [4999.976, 4999.980, 4999.957]))
        assert np.all(np.isclose(ui_units['2'], [6163.200, 2592.000, 2592.000, 2476.800, 2476.800]))

    def test_dac_to_shim_units_terra(self):
        dac_units = {'1': [17729, 18009, 17872], '2': [12500.0] * 5,
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Terra', dac_units)
        assert np.all(np.isclose(ui_units['1'], [3000] * 3))
        assert np.all(np.isclose(ui_units['2'], [9360.0, 4680.0, 4620.0, 4620.0, 4560.0]))

    def test_dac_to_shim_units_0(self):
        dac_units = {'1': [0, 0, 0], '2': [0, 0, 0, 0, 0],
                     'order1_is_valid': True, 'order2_is_valid': True}
        ui_units = dac_to_shim_units('Siemens', 'Prisma_fit', dac_units)
        assert np.all(np.isclose(ui_units['1'], [0, 0, 0]))
        assert np.all(np.isclose(ui_units['2'], [0, 0, 0, 0, 0]))

    def test_dac_to_shim_units_unknown_scanner(self, caplog):
        dac_units = {'1': [14436, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}

        with caplog.at_level(logging.DEBUG, logger.name):
            dac_to_shim_units('Unknown', 'Unknown', dac_units)

        assert "Unknown not implemented or does not include enough metadata information" in caplog.text

    def test_dac_to_shim_units_outside_bounds(self):
        dac_units = {'1': [20000, 14265, 14045], '2': [9998, 9998, 9998, 9998, 9998],
                     'order1_is_valid': True, 'order2_is_valid': True}

        with pytest.raises(ValueError, match="Current shim settings exceed known system limits."):
            dac_to_shim_units('Siemens', 'Prisma_fit', dac_units)


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
