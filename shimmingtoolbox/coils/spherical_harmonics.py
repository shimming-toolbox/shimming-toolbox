#!/usr/bin/python3
# -*- coding: utf-8 -*

from scipy.special import sph_harm
import numpy as np


# def spherical_harmonics(order, X,Y,Z):

phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

m, n = 1, 1

basis = sph_harm(m, n, theta, phi)



