# coding: utf-8

import pytest

import shimmingtoolbox


class TestCore(object):

    # --------------module tests-------------- #
    def test_shimmingtoolbox_module(self):
        assert shimmingtoolbox.__version__
