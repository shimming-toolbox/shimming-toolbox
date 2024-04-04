#!/usr/bin/python3
# -*- coding: utf-8 -*

import math

GYROMAGNETIC_RATIO = 42.577478518e+6  # [Hz/T]


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


def milli_tesla_to_tesla(milli_tesla):
    """ Convert milliTesla to Tesla """
    return milli_tesla * 1e-3


def tesla_to_milli_tesla(tesla):
    """ Convert Tesla to milliTesla """
    return tesla * 1e3


def milli_tesla_to_rad(milli_tesla, dt):
    """ Convert milliTesla to rad """
    tesla = milli_tesla_to_tesla(milli_tesla)
    rad = tesla_to_rad(tesla, dt)
    return rad


def rad_to_milli_tesla(rad, dt):
    """ Convert rad to milliTesla """
    tesla = rad_to_tesla(rad, dt)
    milli_tesla = tesla_to_milli_tesla(tesla)
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
