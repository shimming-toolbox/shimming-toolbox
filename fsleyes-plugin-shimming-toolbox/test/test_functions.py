#!/usr/bin/python3
# -*- coding: utf-8 -*-

from fsleyes_plugin_shimming_toolbox.components.component import get_help_text
from shimmingtoolbox.cli.b0shim import dynamic


def test_get_help_text():
    help_text = get_help_text(dynamic, 'o')
    assert "Directory to output coil text file(s)." == help_text
