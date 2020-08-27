#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import math
from scipy.special import factorial2
from scipy.special import lpmv
from matplotlib.figure import Figure
import os


def number_of_coef(orders):
    out = 0
    for n in orders:
        m = 2 * n + 1
        out += m

    return out


def leg_rec_harmonic_cz(n, m, pos_x, pos_y, pos_z):
    """
    Returns harmonic field for the required solid harmonic addressed by n, m based on the Legendre
    polynomial calculation. Positive m values correspond to the cosine component, negative to the sine. Returned
    fields will eventually follow RRI 's convention pos_... can be both value and vector/matrix
    """

    r2 = pos_x ** 2 + pos_y ** 2 + pos_z ** 2
    r = np.sqrt(r2)
    phi = np.arctan2(pos_y, pos_x)
    cos_theta = np.cos(np.arctan2(np.sqrt(pos_x ** 2 + pos_y ** 2), pos_z))
    # cos_theta = pos_z / r

    if m >= 0:
        c = 1
    else:
        c = 0
        m = -m

    # y_mn = leg_rec(n, m, cos_theta)  # Does the same as lpmv function (Will remove after more tests)
    y_mn = lpmv(m, n, cos_theta)

    rri_norm = math.factorial(n + m + 1) / math.factorial(n - m) / factorial2(2 * m)

    out = (n + m + 1) * (r ** n) * (np.cos(m * phi) * c + np.sin(m * phi) * (1 - c)) * y_mn / rri_norm

    return out


def spherical_harmonics(orders, x, y, z):
    """Return orthonormal spherical harmonic basis set

    Returns an array of spherical harmonic basis fields with the order/degree index along the 4th dimension.

    Args:
        orders (numpy.ndarray):  Degrees of the desired terms in the series expansion, specified as a vector of
                                 non-negative integers (`np.array(range(0, 3))` yields harmonics up to (n-1)-th order).
                                 Must be non negative.
        x (numpy.ndarray): 3-D arrays of grid coordinates
        y (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x)
        z (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x)

    Returns:
        numpy.ndarray: 4d basis set of spherical harmonics with order/degree ordered along 4th dimension

    Examples:
        Initialize grid positions
        >>> [x, y, z] = np.meshgrid(np.array(range(-10, 11)), np.array(range(-10, 11)), np.array(range(-10, 11)),
                        indexing='ij')
        0th-to-2nd order terms inclusive
        >>> orders = np.array(range(0, 3))
        >>> basis = spherical_harmonics(orders, x, y, z)

    Note:
        - basis[:,:,:,1] corresponds to the 0th-order constant term (globally=unity)

        - basis[:,:,:,2:4] to 1st-order linear terms
            - 2: *y*
            - 3: *z*
            - 4: *x*
        - basis[:,:,:,5:9] to 2nd-order terms
            - 5: *xy*
            - 6: *zy*
            - 7: *z2*
            - 8: *zx*
            - 9: *x2y2*

        Based on
            - spherical_harmonics.m by topfer@ualberta.ca
            - calc_spherical_harmonics_arb_points_cz.m by jaystock@nmr.mgh.harvard.edu
    """
    # Check inputs
    if not (x.shape == y.shape == z.shape):
        raise RuntimeError('Input arrays X, Y, and Z must be identically sized')

    if x.ndim == 3:
        grid_size = x.shape
    else:
        raise RuntimeError('Input arrays X, Y, and Z must have 3 dimensions')

    if not np.all(orders >= 0):
        raise RuntimeError('Orders must be positive')

    # Initialize variables
    n_voxels = x.size
    n_orders = orders.size
    n_basis = number_of_coef(orders)
    harm_all = np.zeros([n_voxels, n_basis])

    ii = 0

    # Compute basis
    for iOrder in range(0, n_orders):

        n = orders[iOrder]
        if n == 0:
            m = np.zeros([1], dtype=int)
        else:
            m = np.array(range(-orders[iOrder], orders[iOrder] + 1, 1))

        for mm in range(0, m.size):

            # The first 2 * n(1) + 1 columns of the output correspond to harmonics of order n(1), and the next
            # 2 * n(2) + 1 columns correspond to harmonics of order n(2), etc.
            harm_all[:, ii] = leg_rec_harmonic_cz(n, m[mm], np.reshape(x, [n_voxels]), np.reshape(y, [n_voxels]),
                                                  np.reshape(z, [n_voxels]))

            ii += 1

    # Reshape to initial grid_size
    basis = np.zeros([grid_size[0], grid_size[1], grid_size[2], n_basis])

    for i_basis in range(0, n_basis):
        basis[:, :, :, i_basis] = np.reshape(harm_all[:, i_basis], grid_size)

    return basis
