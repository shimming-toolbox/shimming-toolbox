# coding: utf-8

import pytest
import numpy as np
from shimmingtoolbox.simulate import *

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