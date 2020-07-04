# coding: utf-8

import pytest
import numpy as np
from shimmingtoolbox.simulate import *
from phantominator import shepp_logan

class TestCore(object):
    def setup(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    # --------------class instance tests-------------- #
    def test_initiate_object_is_expected_class(self):

        test_obj = NumericalModel()

        assert isinstance(test_obj, NumericalModel)

    def test_empty_initialization_returns_expected_starting_volume(self):

        test_obj = NumericalModel()

        expected_volume = np.zeros((128, 128))
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    # --------------Shepp-Logan type tests instance test-------------- #
    def test_shepplogan_init_returns_expected_starting_volume(self):
        test_obj = NumericalModel(model='shepp-logan')

        expected_volume = shepp_logan(128)
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepplogan_dims_init_returns_expected_starting_volume(self):
        dims = 256
        test_obj = NumericalModel(model='shepp-logan', num_vox=dims)

        expected_volume = shepp_logan(dims)
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepp_logan_defines_expected_t2star_values(self):
        dims = 256
        test_obj = NumericalModel(model='shepp-logan', num_vox=dims)

        assert np.all(test_obj.volume['T2_star'][abs(test_obj.starting_volume-0.2)<0.001] == test_obj.T2_star['WM'])
        assert np.all(test_obj.volume['T2_star'][abs(test_obj.starting_volume-0.3)<0.001] == test_obj.T2_star['GM'])
        assert np.all(test_obj.volume['T2_star'][abs(test_obj.starting_volume-1)<0.001] == test_obj.T2_star['CSF'])
        assert np.all(test_obj.volume['T2_star'][np.logical_and((abs(test_obj.starting_volume)<0.0001), test_obj.starting_volume!=0)] == test_obj.T2_star['WM']/2)
        assert np.all(test_obj.volume['T2_star'][abs(test_obj.starting_volume-0.1)<0.001] == (test_obj.T2_star['GM']+test_obj.T2_star['WM'])/2)
        assert np.all(test_obj.volume['T2_star'][abs(test_obj.starting_volume-0.4)<0.001] == test_obj.T2_star['GM'] * 1.5)

    def test_shepp_logan_defines_expected_magnitude_values(self):
        dims = 256
        test_obj = NumericalModel(model='shepp-logan', num_vox=dims)

        assert np.all(test_obj.volume['proton_density'][abs(test_obj.starting_volume-0.2)<0.001] == test_obj.proton_density['WM'])
        assert np.all(test_obj.volume['proton_density'][abs(test_obj.starting_volume-0.3)<0.001] == test_obj.proton_density['GM'])
        assert np.all(test_obj.volume['proton_density'][abs(test_obj.starting_volume-1)<0.001] == test_obj.proton_density['CSF'])
        assert np.all(test_obj.volume['proton_density'][np.logical_and((abs(test_obj.starting_volume)<0.0001), test_obj.starting_volume!=0)] == test_obj.proton_density['WM']/2)
        assert np.all(test_obj.volume['proton_density'][abs(test_obj.starting_volume-0.1)<0.001] == (test_obj.proton_density['GM']+test_obj.proton_density['WM'])/2)
        assert np.all(test_obj.volume['proton_density'][abs(test_obj.starting_volume-0.4)<0.001] == test_obj.proton_density['GM'] * 1.5)
