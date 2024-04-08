#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Test module to test the conversion functions in conversion.py"""

import math

from shimmingtoolbox.conversion import (hz_to_rad_per_sec, rad_per_sec_to_hz, hz_to_tesla, tesla_to_hz,
                                        rad_per_sec_to_rad, rad_to_rad_per_sec, hz_to_rad, rad_to_hz, tesla_to_rad,
                                        rad_to_tesla, milli_tesla_to_rad, rad_to_milli_tesla, gauss_to_rad,
                                        rad_to_gauss, hz_to_micro_tesla, micro_tesla_to_hz,
                                        hz_per_cm2_to_micro_tesla_per_m2, micro_tesla_per_m2_to_hz_per_cm2,
                                        hz_per_cm_to_micro_tesla_per_m, micro_tesla_per_m_to_hz_per_cm,
                                        metric_unit_to_metric_unit, unit_per_metric_unit_to_unit_per_metric_unit)


def test_hz_to_rad_per_sec():
    assert hz_to_rad_per_sec(1) == 2 * math.pi
    assert hz_to_rad_per_sec(0) == 0


def test_rad_per_sec_to_hz():
    assert rad_per_sec_to_hz(2 * math.pi) == 1
    assert rad_per_sec_to_hz(0) == 0


def test_hz_to_tesla():
    assert math.isclose(hz_to_tesla(42.577478518e+6), 1)
    assert hz_to_tesla(0) == 0


def test_tesla_to_hz():
    assert math.isclose(tesla_to_hz(1), 42.577478518e+6)
    assert tesla_to_hz(0) == 0


def test_hz_to_micro_tesla():
    assert math.isclose(hz_to_micro_tesla(42.577478518), 1)


def test_micro_tesla_to_hz():
    assert math.isclose(micro_tesla_to_hz(1), 42.577478518)


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
    assert math.isclose(tesla_to_rad(1, 0.1), 0.2 * math.pi * 42.577478518e+6)
    assert tesla_to_rad(0, 0.1) == 0


def test_rad_to_tesla():
    assert math.isclose(rad_to_tesla(2 * math.pi, 0.1), 10 / 42.577478518e+6)
    assert rad_to_tesla(0, 0.1) == 0


def test_milli_tesla_to_rad():
    assert math.isclose(milli_tesla_to_rad(1, 0.1), 0.2e-3 * math.pi * 42.577478518e+6)
    assert milli_tesla_to_rad(0, 0.1) == 0


def test_rad_to_milli_tesla():
    assert math.isclose(rad_to_milli_tesla(2 * math.pi, 0.1), 10e3 / 42.577478518e+6)
    assert rad_to_milli_tesla(0, 0.1) == 0


def test_gauss_to_rad():
    assert math.isclose(gauss_to_rad(1, 0.1), 0.2e-4 * math.pi * 42.577478518e+6)
    assert gauss_to_rad(0, 0.1) == 0


def test_rad_to_gauss():
    assert math.isclose(rad_to_gauss(2 * math.pi, 0.1), 10e4 / 42.577478518e+6)
    assert rad_to_gauss(0, 0.1) == 0


def test_cm_to_m():
    assert metric_unit_to_metric_unit(500, 'c', '') == 5


def test_m_to_cm():
    assert metric_unit_to_metric_unit(5, 'b', 'c') == 500


def test_cm2_to_m2():
    assert metric_unit_to_metric_unit(50000, 'c', '', 2) == 5


def test_m2_to_cm2():
    assert metric_unit_to_metric_unit(5, '', 'c', 2) == 50000


def test_unit_per_cm_to_unit_per_m():
    assert unit_per_metric_unit_to_unit_per_metric_unit(1, 'c', '') == 1e2


def test_unit_per_m_to_unit_per_cm():
    assert unit_per_metric_unit_to_unit_per_metric_unit(1e2, '', 'c') == 1


def test_hz_per_cm_to_micro_tesla_per_m():
    assert math.isclose(hz_per_cm_to_micro_tesla_per_m(0.42577478518), 1)


def test_micro_tesla_per_m_to_hz_per_cm():
    assert math.isclose(micro_tesla_per_m_to_hz_per_cm(1), 0.42577478518)


def test_unit_per_cm2_to_unit_per_m2():
    assert unit_per_metric_unit_to_unit_per_metric_unit(1, 'c', '', 2) == 1e4


def test_unit_per_m2_to_unit_per_cm2():
    assert unit_per_metric_unit_to_unit_per_metric_unit(1e4, '', 'c', 2) == 1


def test_hz_per_cm2_to_micro_tesla_per_m2():
    assert math.isclose(hz_per_cm2_to_micro_tesla_per_m2(0.0042577478518), 1)


def test_micro_tesla_per_m2_to_hz_per_cm2():
    assert math.isclose(micro_tesla_per_m2_to_hz_per_cm2(1), 0.0042577478518)
