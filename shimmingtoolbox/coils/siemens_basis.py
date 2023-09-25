#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics


X = 0
Y = 1
Z = 2
ZX = 4
XY = 7
GYROMAGNETIC_RATIO = 42.5774785178325552
order_indexes = {1: [0, 3], 2: [3, 8]}

def _reorder_to_siemens(spher_harm, orders):
    """
    Reorder 1st - 2nd order basis terms along 4th dim. From
    1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to
    2. X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with Siemens shims)

    Args:
        spher_harm (numpy.ndarray): 4d basis set of spherical harmonics with order/degree ordered along 4th
                                    dimension. ``spher_harm.shape[3]`` must equal 8.

    Returns:
        numpy.ndarray: 4d basis set of spherical harmonics ordered following siemens convention
    """
    print(f"orders: {orders}")
    if orders == (1, 2):
        if spher_harm.shape[3] != 8:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 8")

        reordered = spher_harm[:, :, :, [2, 0, 1, 5, 6, 4, 7, 3]]

    if orders == (2,):
        if spher_harm.shape[3] != 5:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 5")

        reordered = spher_harm[:, :, :, [2, 3, 1, 4, 0]]

    if orders == (1,):
        if spher_harm.shape[3] != 3:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 3")

        reordered = spher_harm[:, :, :, [2, 0, 1]]

    return reordered


def _get_scaling_factors(orders):
    """
    Get scaling factors for the 8 terms to apply to the (Siemens-reordered) 1st + 2nd order spherical harmonic
    fields for rescaling field terms as "shim reference maps" in units of Hz/unit-shim:

    Gx, Gy, and Gz should yield 1 micro-T of field shift per metre: equivalently, 0.042576 Hz/mm

    2nd order terms should yield 1 micro-T of field shift per metre-squared: equivalently, 0.000042576 Hz/mm^2

    Gist: given the stated nominal values, we can pick several arbitrary reference positions around the
    origin/isocenter at which we know what the field *should* be, and use that to calculate the appropriate scaling
    factor.

    Returns:
         numpy.ndarray:  1D vector of ``scaling_factors``

    NOTE: The method has been worked out empirically and has only been tested for 2 Siemens Prisma systems.
    E.g.re: Y, Z, ZX, and XY terms, their polarity had to be flipped relative to the form given by
    ``shimmingtoolbox.coils.spherical_harmonics``.
    """

    [x_iso, y_iso, z_iso] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                        indexing='xy')
    orders_array = np.array([order for order in orders if order != 0])
    sh = spherical_harmonics(orders_array, x_iso, y_iso, z_iso)
    sh = _reorder_to_siemens(sh, orders)

    n_channels = sh.shape[3]
    scaling_factors = np.zeros(n_channels)

    # indices of reference positions for normalization:
    i_x1 = np.nonzero((x_iso == 1) & (y_iso == 0) & (z_iso == 0))
    i_y1 = np.nonzero((x_iso == 0) & (y_iso == 1) & (z_iso == 0))
    i_z1 = np.nonzero((x_iso == 0) & (y_iso == 0) & (z_iso == 1))

    i_x1z1 = np.nonzero((x_iso == 1) & (y_iso == 0) & (z_iso == 1))
    i_y1z1 = np.nonzero((x_iso == 0) & (y_iso == 1) & (z_iso == 1))
    i_x1y1 = np.nonzero((x_iso == 1) & (y_iso == 1) & (z_iso == 0))

    # order the reference indices like the sh field terms
    i_ref = [i_x1, i_y1, i_z1, i_z1, i_x1z1, i_y1z1, i_x1, i_x1y1]

    # distance from iso/origin to adopted reference point[units: mm]
    r = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2)]

    # scaling:
    orders = [1, 1, 1, 2, 2, 2, 2, 2]

    for i_ch in range(0, n_channels):
        field = sh[:, :, :, i_ch]
        scaling_factors[i_ch] = GYROMAGNETIC_RATIO * ((r[i_ch] * 0.001) ** orders[i_ch]) / field[i_ref[i_ch]][0]

    return scaling_factors


def siemens_basis(x, y, z, orders=(1, 2)):
    """
    The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. *Following Siemens convention*,
    ``basis`` is then:

        - Reordered along the 4th dimension as *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*

        - Rescaled to Hz/unit-shim, where "unit-shim" refers to the measure displayed in the Adjustments card of the
          Syngo console UI, namely:

            - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
            - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)

    The returned ``basis`` is thereby in the form of ideal "shim reference maps", ready for optimization.

    Args:
        x (numpy.ndarray): 3-D arrays of grid coordinates, "Left->Right" grid coordinates in the patient coordinate
                           system (i.e. NIfTI reference (RAS), units of mm)
        y (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Posterior->Anterior" grid coordinates in
                           the patient coordinate system (i.e. NIfTI reference (RAS), units of mm)
        z (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Inferior->Superior" grid coordinates in
                           the patient coordinate system (i.e. NIfTI reference, units of mm)
        orders (tuple): Degrees of the desired terms in the series expansion, specified as a vector of non-negative
                        integers (``(0:1:n)`` yields harmonics up to n-th order, implemented 1st and 2nd order)

    Returns:
        numpy.ndarray: 4-D array of spherical harmonic basis fields

    NOTES:
        For now, ``orders`` is, in fact, as default [1:2]—which is suitable for the Prisma (presumably other
        Siemens systems as well) however, the 3rd-order shims of the Terra should ultimately be accommodated too.
        (Requires checking the Adjustments/Shim card to see what the corresponding terms and values actually are). So,
        ``basis`` will return with 8 terms along the 4th dim if using the 1st and 2nd order.
    """

    # Check inputs
    if not (x.ndim == y.ndim == z.ndim == 3):
        raise RuntimeError("Input arrays X, Y, and Z must be 3d")

    if not (x.shape == y.shape == z.shape):
        raise RuntimeError("Input arrays X, Y, and Z must be identically sized")

    if max(orders) >= 3:
        raise NotImplementedError("Spherical harmonics not implemented for order 3 and up")

    # Create spherical harmonics from first to second order
    all_orders = np.array([order for order in orders if order != 0])
    spher_harm = spherical_harmonics(all_orders, x, y, z)
    # Reorder according to siemens convention: X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
    reordered_spher = _reorder_to_siemens(spher_harm, orders)

    # scale according to
    # - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
    # - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)
    scaling_factors = _get_scaling_factors(orders)
    scaled = np.zeros_like(reordered_spher)
    for i_channel in range(0, spher_harm.shape[3]):
        scaled[:, :, :, i_channel] = scaling_factors[i_channel] * reordered_spher[:, :, :, i_channel]

    # Patch to make orders work. A better implementation would be to refactor _get_scaling_factors and
    # _reorder_to_siemens
    range_per_order = {}
    index = 0
    for order in orders:
        range_per_order[order] = list(range(index, index+(order*2 +1)))
        index += order*2 + 1
    print(range_per_order)
    # range_per_order = {1: list(range(3)), 2: list(range(3, 8))}
    length_dim3 = np.sum([len(values) for key, values in range_per_order.items() if key in orders])
    output = np.zeros(scaled[..., 0].shape + (length_dim3,), dtype=scaled.dtype)
    start_index = 0
    for order in orders:
        end_index = start_index + len(range_per_order[order])
        print(f"scaled: {scaled.shape}")
        print(f"output: {output.shape}")
        output[..., start_index:end_index] = scaled[..., range_per_order[order]]
        # prep next iteration
        start_index = end_index

    return output
