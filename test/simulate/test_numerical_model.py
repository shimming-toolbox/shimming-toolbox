# coding: utf-8

import pytest
from shimmingtoolbox.simulate import *

class TestCore(object):
    def setup(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    # --------------class tests-------------- #
    def test_object_creation(self):

        sim_obj = NumericalModel()

        assert isinstance(sim_obj, NumericalModel)
