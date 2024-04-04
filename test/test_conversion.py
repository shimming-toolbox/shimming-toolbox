#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Test module to test the conversion functions in conversion.py"""

import math

from shimmingtoolbox.conversion import (hz_to_rad_per_sec, rad_per_sec_to_hz, hz_to_tesla, tesla_to_hz,
                                        rad_per_sec_to_rad, rad_to_rad_per_sec, hz_to_rad, rad_to_hz, tesla_to_rad,
                                        rad_to_tesla, milli_tesla_to_rad, rad_to_milli_tesla, gauss_to_rad,
                                        rad_to_gauss)


def test_hz_to_rad_per_sec():
    assert hz_to_rad_per_sec(1) == 2 * math.pi
    assert hz_to_rad_per_sec(0) == 0


def test_rad_per_sec_to_hz():
    assert rad_per_sec_to_hz(2 * math.pi) == 1
    assert rad_per_sec_to_hz(0) == 0


def test_hz_to_tesla():
    assert hz_to_tesla(42.577478518e+6) == 1
    assert hz_to_tesla(0) == 0


def test_tesla_to_hz():
    assert tesla_to_hz(1) == 42.577478518e+6
    assert tesla_to_hz(0) == 0


def test_rad_per_sec_to_rad():
    assert rad_per_sec_to_rad(1, 0.1) == 0.1
    assert rad_per_sec_to_rad(0, 0.1) == 0


def test_rad_to_rad_per_sec():
    assert rad_to_rad_per_sec(1, 0.1) == 10
    assert rad_to_rad_per_sec(0, 0.1) == 0


def test_hz_to_rad():
    assert hz_to_rad(1, 0.1) == 0.2 * math.pi
    assert hz_to_rad(0, 0.1) == 0


def test_rad_to_hz():
    assert rad_to_hz(2 * math.pi, 0.1) == 10
    assert rad_to_hz(0, 0.1) == 0


def test_tesla_to_rad():
    assert tesla_to_rad(1, 0.1) == 0.2 * math.pi * 42.577478518e+6
    assert tesla_to_rad(0, 0.1) == 0


def test_rad_to_tesla():
    assert rad_to_tesla(2 * math.pi, 0.1) == 10 / 42.577478518e+6
    assert rad_to_tesla(0, 0.1) == 0


def test_milli_tesla_to_rad():
    assert math.isclose(milli_tesla_to_rad(1, 0.1), 0.2e-3 * math.pi * 42.577478518e+6)
    assert milli_tesla_to_rad(0, 0.1) == 0


def test_rad_to_milli_tesla():
    assert rad_to_milli_tesla(2 * math.pi, 0.1) == 10e3 / 42.577478518e+6
    assert rad_to_milli_tesla(0, 0.1) == 0


def test_gauss_to_rad():
    assert gauss_to_rad(1, 0.1) == 0.2e-4 * math.pi * 42.577478518e+6
    assert gauss_to_rad(0, 0.1) == 0


def test_rad_to_gauss():
    assert rad_to_gauss(2 * math.pi, 0.1) == 10e4 / 42.577478518e+6
    assert rad_to_gauss(0, 0.1) == 0
