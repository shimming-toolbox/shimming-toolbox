#!/usr/bin/python3
# -*- coding: utf-8 -*

import math

GYROMAGNETIC_RATIO = 42.5774785178325552e+6  # [Hz/T]
METRIC_PREFIXES = {
    'n': 1e-9,  # 'nano'
    'u': 1e-6,  # 'micro'
    'm': 1e-3,  # 'milli'
    'c': 1e-2,  # 'centi'
    '': 1,  # 'base'
    'b': 1,  # 'base'
    'h': 1e2,  # 'hecto'
    'k': 1e3,  # 'kilo'
    'M': 1e6,  # 'mega'
    'G': 1e9  # 'giga'
}


def metric_unit_to_metric_unit(x, prefix_in, prefix_out, power=1):
    """ Convert units with metric prefixes to other metric prefixes (i.e. T to uT, 1/cm^2 to 1/m^2).

    Args:
        x: Float or array of floats
        prefix_in (str): Prefix of the input unit
        prefix_out (str): Prefix of the output unit
        power (int): Power of the unit, use 2 for squared units (i.e. m^2), to convert x/cm to x/m, use a negative power

    Returns:
        Float or array of floats converted
    """
    if prefix_in not in METRIC_PREFIXES or prefix_out not in METRIC_PREFIXES:
        raise ValueError('Invalid prefix')
    factor = (METRIC_PREFIXES[prefix_in] / METRIC_PREFIXES[prefix_out]) ** power
    return x * factor


def hz_to_rad_per_sec(hz):
    """ Convert Hz to rad/s """
    return hz * 2 * math.pi


def rad_per_sec_to_hz(rad_per_sec):
    """ Convert rad/s to Hz """
    return rad_per_sec / (2 * math.pi)


def hz_to_tesla(hz):
    """ Convert Hz to Tesla """
    return hz / GYROMAGNETIC_RATIO


def tesla_to_hz(tesla):
    """ Convert Tesla to Hz """
    return tesla * GYROMAGNETIC_RATIO


def hz_to_micro_tesla(hz):
    """ Convert Hz to microTesla """
    tesla = hz_to_tesla(hz)
    micro_tesla = metric_unit_to_metric_unit(tesla, '', 'u')
    return micro_tesla


def micro_tesla_to_hz(micro_tesla):
    """ Convert microTesla to Hz """
    tesla = metric_unit_to_metric_unit(micro_tesla, 'u', '')
    hz = tesla_to_hz(tesla)
    return hz


def rad_per_sec_to_rad(rad_per_sec, dt):
    """ Convert rad/s to rad """
    return rad_per_sec * dt


def rad_to_rad_per_sec(rad, dt):
    """ Convert rad to rad/s """
    return rad / dt


def hz_to_rad(hz, dt):
    """ Convert Hz to rad """
    rad_per_sec = hz_to_rad_per_sec(hz)
    rad = rad_per_sec_to_rad(rad_per_sec, dt)
    return rad


def rad_to_hz(rad, dt):
    """ Convert rad to Hz """
    rad_per_sec = rad_to_rad_per_sec(rad, dt)
    hz = rad_per_sec_to_hz(rad_per_sec)
    return hz


def tesla_to_rad(tesla, dt):
    """ Convert Tesla to rad """
    hz = tesla_to_hz(tesla)
    rad = hz_to_rad(hz, dt)
    return rad


def rad_to_tesla(rad, dt):
    """ Convert rad to Tesla """
    hz = rad_to_hz(rad, dt)
    tesla = hz_to_tesla(hz)
    return tesla


def milli_tesla_to_rad(milli_tesla, dt):
    """ Convert milliTesla to rad """
    tesla = metric_unit_to_metric_unit(milli_tesla, 'm', '')
    rad = tesla_to_rad(tesla, dt)
    return rad


def rad_to_milli_tesla(rad, dt):
    """ Convert rad to milliTesla """
    tesla = rad_to_tesla(rad, dt)
    milli_tesla = metric_unit_to_metric_unit(tesla, '', 'm')
    return milli_tesla


def gauss_to_tesla(gauss):
    """ Convert Gauss to Tesla """
    return gauss * 1e-4


def tesla_to_gauss(tesla):
    """ Convert Tesla to Gauss """
    return tesla * 1e4


def gauss_to_rad(gauss, dt):
    """ Convert Gauss to rad """
    tesla = gauss_to_tesla(gauss)
    rad = tesla_to_rad(tesla, dt)
    return rad


def rad_to_gauss(rad, dt):
    """ Convert rad to Gauss """
    tesla = rad_to_tesla(rad, dt)
    gauss = tesla_to_gauss(tesla)
    return gauss


def hz_per_cm_to_micro_tesla_per_m(hz_per_cm):
    """ Convert Hz/cm to microTesla/m """
    micro_tesla_per_cm = hz_to_micro_tesla(hz_per_cm)
    micro_tesla_per_m = metric_unit_to_metric_unit(micro_tesla_per_cm, 'c', '', power=-1)
    return micro_tesla_per_m


def micro_tesla_per_m_to_hz_per_cm(micro_tesla_per_m):
    """ Convert microTesla/m to Hz/cm """
    micro_tesla_per_cm = metric_unit_to_metric_unit(micro_tesla_per_m, '', 'c', power=-1)
    hz_per_cm = micro_tesla_to_hz(micro_tesla_per_cm)
    return hz_per_cm


def hz_per_cm2_to_micro_tesla_per_m2(hz_per_cm2):
    """ Convert Hz/cm^2 to microTesla/m^2 """
    micro_tesla_per_cm2 = hz_to_micro_tesla(hz_per_cm2)
    micro_tesla_per_m2 = metric_unit_to_metric_unit(micro_tesla_per_cm2, 'c', '', power=-2)
    return micro_tesla_per_m2


def micro_tesla_per_m2_to_hz_per_cm2(micro_tesla_per_m2):
    """ Convert microTesla/m^2 to Hz/cm^2 """
    micro_tesla_per_cm2 = metric_unit_to_metric_unit(micro_tesla_per_m2, '', 'c', -2)
    hz_per_cm2 = micro_tesla_to_hz(micro_tesla_per_cm2)
    return hz_per_cm2
