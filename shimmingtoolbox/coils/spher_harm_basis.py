#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import numpy as np

from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics

logger = logging.getLogger(__name__)

GYROMAGNETIC_RATIO = 42.5774785178325552  # [MHz/T]
ORDER_INDEXES = {1: 3, 2: 5}

def siemens_basis(x, y, z, orders=(1, 2), shim_cs='LAI'):
    """
    The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. *Following Siemens convention*,
    ``basis`` is then:

        - Rescaled to Hz/unit-shim, where "unit-shim" refers to the measure displayed in the Adjustments card of the
          Syngo console UI, namely:

            - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
            - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)

        - Reordered along the 4th dimension as *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*

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
        shim_cs (str): Coordinate system of the shims. Letter 1 'R' or 'L', letter 2 'A' or 'P', letter 3 'S' or 'I'.

    Returns:
        numpy.ndarray: 4-D array of spherical harmonic basis fields

    NOTES:
        For now, ``orders`` is, in fact, as default [1:2]—which is suitable for the Prisma (presumably other
        Siemens systems as well) however, the 3rd-order shims of the Terra should ultimately be accommodated too.
        (Requires checking the Adjustments/Shim card to see what the corresponding terms and values actually are). So,
        ``basis`` will return with 8 terms along the 4th dim if using the 1st and 2nd order.
    """

    # Check inputs
    _check_basis_inputs(x, y, z, orders)

    # Create spherical harmonics from first to second order
    all_orders = np.array([order for order in orders if order != 0])
    spher_harm = scaled_spher_harm(x, y, z, all_orders, shim_cs=shim_cs)

    # Reorder according to siemens convention: X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
    reordered_spher = _reorder_to_siemens(spher_harm, orders)

    # Select order
    range_per_order = {}
    index = 0
    for order in orders:
        range_per_order[order] = list(range(index, index + ORDER_INDEXES[order]))
        index += ORDER_INDEXES[order]

    # range_per_order = {1: list(range(3)), 2: list(range(3, 8))}
    length_dim3 = np.sum([len(values) for key, values in range_per_order.items() if key in orders])
    output = np.zeros(reordered_spher[..., 0].shape + (length_dim3,), dtype=reordered_spher.dtype)
    start_index = 0
    for order in orders:
        end_index = start_index + len(range_per_order[order])
        output[..., start_index:end_index] = reordered_spher[..., range_per_order[order]]
        # prep next iteration
        start_index = end_index

    return output


def _reorder_to_siemens(spher_harm, orders):
    """
    Reorder 1st - 2nd order coefficients along the last dim. From
    1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to
    2. X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with Siemens shims)

    Args:
        spher_harm (numpy.ndarray): Coefficients with the last dimensions representing the different order channels.
                                    ``spher_harm.shape[-1]`` must equal 8.
        orders (tuple): Spherical harmonics orders to use

    Returns:
        numpy.ndarray: Coefficients ordered following Siemens convention
    """
    if orders == (1, 2):
        if spher_harm.shape[-1] != 8:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 8")

        reordered = spher_harm[..., [2, 0, 1, 5, 6, 4, 7, 3]]

    if orders == (2,):
        if spher_harm.shape[-1] != 5:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 5")

        reordered = spher_harm[..., [2, 3, 1, 4, 0]]

    if orders == (1,):
        if spher_harm.shape[-1] != 3:
            raise RuntimeError(f"Input arrays should have 4th dimension's shape equal to 3 {spher_harm.shape}")

        reordered = spher_harm[..., [2, 0, 1]]

    return reordered


def ge_basis(x, y, z, orders=(1, 2), shim_cs='LPI'):
    """
    The function first wraps ``shimmingtoolbox.coils.spher_harm_basis.scaled_spher_harm`` to generate 1st and 2nd
    order spherical harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``.
    *Following GE convention*, ``basis`` is then:

        - Reordered along the 4th dimension as *x, y, z, xy, zy, zx, X2-Y2, z2*

        - Rescaled:

            - 1 G/cm for *X,Y,Z* gradients (= Hz/mm)
            - Hz/mm^2 / 1 mA for the 2nd order terms (See details below for the different channels)

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
        shim_cs (str): Coordinate system of the shims. Letter 1 'R' or 'L', letter 2 'A' or 'P', letter 3 'S' or 'I'.

    Returns:
        numpy.ndarray: 4-D array of spherical harmonic basis fields

    NOTES:
        For now, ``orders`` is, in fact, as default [1:2]. So, ``basis`` will return with 8 terms along the
        4th dim if using the 1st and 2nd order.
        """

    # Check inputs
    _check_basis_inputs(x, y, z, orders)

    # Create spherical harmonics from first to second order
    all_orders = np.array([order for order in orders if order != 0])
    spher_harm = scaled_spher_harm(x, y, z, all_orders, shim_cs=shim_cs)

    # The following matrix (8 x 5) refers to the following:
    #  \  xy, zy, zx, XY, z2
    # xy
    # zy
    # zx
    # XY
    # z2
    # x
    # y
    # z
    # B0

    # Order 2: [Hz/cm2/A]
    # Order 1: [Hz/cm/A]
    # Order 0: [Hz/A]

    # e.g. 1A in xy will produce 1.8367Hz/cm2 in xy, -0.0018785Hz/cm2 in zy
    order2_to_order2 = np.array([[1.8367, -0.0018785, -0.0038081, -0.001403, -0.00029865],
                                 [-0.0067895, 2.2433, 0.0086222, 0.008697, -0.0091463],
                                 [-0.0046842, 0.0030595, 2.2174, -0.0073788, 0.013379],
                                 [7.8099e-05, 0.0041857, -0.0044671, 0.90317, -0.0079819],
                                 [-0.00060671, -0.0077883, -0.010489, -0.0036487, 2.0056]])
    # [0.096079, -0.061232, 0.37826, -0.11422, -0.55906],
    # [-0.077678, 0.69811, 0.04663, -0.16589, -0.10913],
    # [-0.34644, 0.15591, -0.13374, -0.34059, 1.7438],
    # [0.85228, 2.3916, -0.10486, 0.48776, -305.75]])

    # Reorder according to GE convention: x, y, z, xy, zy, zx, X2-Y2, z2
    reordered_spher = _reorder_to_ge(spher_harm, orders)

    # Scale
    # Hz/cm2/A, -> uT/m2/A = order2_to_order2 * 1e6 * (100 ** 2) / (GYROMAGNETIC_RATIO * 1e6)
    # = order2_to_order2 * (100 ** 2) / GYROMAGNETIC_RATIO
    orders_to_order2_uT = order2_to_order2 * (100 ** 2) / GYROMAGNETIC_RATIO


    # Order 2
    scaled = np.zeros_like(reordered_spher)
    index = 0
    if 1 in orders:

        # Rescale to unit-shims that are G/cm
        # They are currently in uT/m
        # 1G = 1e-4T, 1T = 1e4G
        # uT/m --> G/cm = reordered_spher * (1/1e6) * 1e4  * 100 = reordered_sphere
        scaled[..., 0:ORDER_INDEXES[1]] = reordered_spher[..., 0:ORDER_INDEXES[1]]
        index += ORDER_INDEXES[1]

    if 2 in orders:
        for i_channel in range(index, index + ORDER_INDEXES[2]):
            # Since reordered_spher contains the values of 1uT/m^2 in Hz/mm^2. We simply multiply by the amount of
            # uT/m^2 / A
            # This gives us a value in Hz/mm^2 / A which we need to modify to Hz/mm^2 / mA
            scaled[..., i_channel] = np.matmul(reordered_spher[..., index:index + ORDER_INDEXES[2]],
                                               orders_to_order2_uT[i_channel - index, :]) / 1000

    # Output according to the specified order
    range_per_order = {}
    index = 0
    for order in orders:
        range_per_order[order] = list(range(index, index + ORDER_INDEXES[order]))
        index += ORDER_INDEXES[order]

    length_dim3 = np.sum([len(values) for key, values in range_per_order.items() if key in orders])
    output = np.zeros(scaled[..., 0].shape + (length_dim3,), dtype=scaled.dtype)
    start_index = 0
    for order in orders:
        end_index = start_index + len(range_per_order[order])
        output[..., start_index:end_index] = scaled[..., range_per_order[order]]
        # prep next iteration
        start_index = end_index

    return output


def _reorder_to_ge(spher_harm, orders):
    """
    Reorder 1st - 2nd order coefficients along the last dim. From
    1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to
    2. x, y, z, xy, zy, zx, X2 - Y2, z2 (in line with GE shims)

    Args:
        spher_harm (numpy.ndarray): Coefficients with the last dimensions representing the different order channels.
                                    ``spher_harm.shape[-1]`` must equal 8.

    Returns:
        numpy.ndarray: Coefficients ordered following GE convention
    """

    if orders == (1, 2):
        if spher_harm.shape[-1] != 8:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 8")
        reordered = spher_harm[..., [2, 0, 1, 3, 4, 6, 7, 5]]

    if orders == (2,):
        if spher_harm.shape[-1] != 5:
            raise RuntimeError("Input arrays should have 4th dimension's shape equal to 5")
        reordered = spher_harm[..., [0, 1, 3, 4, 2]]

    if orders == (1,):
        if spher_harm.shape[-1] != 3:
            raise RuntimeError(f"Input arrays should have 4th dimension's shape equal to 3 {spher_harm.shape}")
        reordered = spher_harm[..., [2, 0, 1]]

    return reordered


def scaled_spher_harm(x, y, z, all_orders, shim_cs='ras'):
    """ The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. It is then:

        - Rescaled to 1uT/m or 1uT/m^2 in units of Hz/mm or Hz/mm^2:

            - 1 micro-T/m for *X,Y,Z* gradients(= 0.042576 Hz/mm)
            - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)

    Args:
        x (numpy.ndarray): 3-D arrays of grid coordinates, "Left->Right" grid coordinates in the patient coordinate
                           system (i.e. NIfTI reference (RAS), units of mm)
        y (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Posterior->Anterior" grid coordinates in
                           the patient coordinate system (i.e. NIfTI reference (RAS), units of mm)
        z (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Inferior->Superior" grid coordinates in
                           the patient coordinate system (i.e. NIfTI reference, units of mm)
        orders (tuple): Degrees of the desired terms in the series expansion, specified as a vector of non-negative
                        integers (``(0:1:n)`` yields harmonics up to n-th order, implemented 1st and 2nd order)
        shim_cs (str): Coordinate system of the shims. Letter 1 'R' or 'L', letter 2 'A' or 'P', letter 3 'S' or 'I'.

    Returns:
        numpy.ndarray: 4d basis set of spherical harmonics scaled
    """
    # Check inputs
    _check_basis_inputs(x, y, z, all_orders)

    # Create spherical harmonics from first to second order
    spher_harm = spherical_harmonics(all_orders, x, y, z)
    # 1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics)

    # spher_harm_cs = _reorder_to_siemens(spher_harm, tuple(all_orders))
    spher_harm_cs = get_flip_matrix(shim_cs, all_orders) * spher_harm

    # scale according to
    # - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
    # - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)
    scaling_factors = _get_scaling_factors()
    scaled = np.zeros_like(spher_harm_cs)
    index = 0
    for order in all_orders:
        scaled[:, :, :, index:index + ORDER_INDEXES[order]] = np.array(scaling_factors[order]).squeeze() \
            * spher_harm_cs[:, :, :, index:index + ORDER_INDEXES[order]]
        index += ORDER_INDEXES[order]

    # 1 uT/m, 1 uT/m2 in Hz/mm, Hz/mm2
    return scaled


def _get_scaling_factors():
    """
    Get scaling factors for the 8 terms to apply to the 1st + 2nd order spherical harmonic
    fields for rescaling them to 1 uT/unit-shim in units of Hz/mm:

    Gx, Gy, and Gz should yield 1 micro-T of field shift per metre: equivalently, 0.042576 Hz/mm

    2nd order terms should yield 1 micro-T of field shift per metre-squared: equivalently, 0.000042576 Hz/mm^2

    Gist: given the stated nominal values, we can pick several arbitrary reference positions around the
    origin/isocenter at which we know what the field *should* be, and use that to calculate the appropriate scaling
    factor.

    Returns:
         numpy.ndarray:  1D vector of ``scaling_factors``
    """

    [x_iso, y_iso, z_iso] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                        indexing='xy')
    orders = np.array(range(1, 3))
    sh = spherical_harmonics(orders, x_iso, y_iso, z_iso)

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
    # 1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to
    i_ref = [i_y1, i_z1, i_x1, i_x1y1, i_y1z1, i_z1, i_x1z1, i_x1]

    # distance from iso/origin to adopted reference point[units: mm]
    r = [1, 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1]

    # scaling:
    orders = [1, 2]
    scaling_factors = {}
    index = 0
    for order in orders:
        scaling_factors[order] = [GYROMAGNETIC_RATIO * ((r[i_ch] * 0.001) ** order)
                                  / sh[:, :, :, i_ch][i_ref[i_ch]] for i_ch in range(index, index + ORDER_INDEXES[order])]
        index += ORDER_INDEXES[order]
    return scaling_factors


def _check_basis_inputs(x, y, z, orders):
    # Check inputs
    if not (x.ndim == y.ndim == z.ndim == 3):
        raise RuntimeError("Input arrays X, Y, and Z must be 3d")

    if not (x.shape == y.shape == z.shape):
        raise RuntimeError("Input arrays X, Y, and Z must be identically sized")

    if max(orders) >= 3:
        raise NotImplementedError("Spherical harmonics not implemented for order 3 and up")


def get_flip_matrix(shim_cs='ras', orders=(1,2), manufacturer=None):
    """
    Return a matrix to flip the spherical harmonics basis set from ras to the desired coordinate system.

    Args:
        shim_cs (str): Coordinate system of the shim basis set. Default is RAS.
        xyz (bool): If True, return the matrix to flip for xyz only.
        manufacturer (str): Manufacturer of the scanner. The flipping matrix is different for each manufacturer.
                            If None is selected, it will output according to
                            ``shimmingtoolbox.coils.spherical_harmonics``. Possible values: SIEMENS, GE.

    Returns:
        numpy.ndarray: Matrix (len: 8) to flip the spherical harmonics basis set from ras to the desired coordinate
                       system. Output is a 1D vector of ``flip_matrix`` for the following:
                       Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2. if xyz is True, output X, Y, Z only in this order.
    """
    xyz_cs = [1, 1, 1]

    shim_cs = shim_cs.upper()
    if (len(shim_cs) != 3) or \
            (shim_cs[0] not in ['R', 'L']) or (shim_cs[1] not in ['A', 'P']) or (shim_cs[2] not in ['S', 'I']):
        raise ValueError("Unknown coordinate system: {}".format(shim_cs))

    if shim_cs[0] == 'L':
        xyz_cs[0] = -1
    if shim_cs[1] == 'P':
        xyz_cs[1] = -1
    if shim_cs[2] == 'I':
        xyz_cs[2] = -1

    temp_list_out = []
    if 1 in orders:
        # Y, Z, X
        out_matrix = temp_list_out.extend([xyz_cs[1], xyz_cs[2], xyz_cs[0]])
    if 2 in orders:
        # XY, ZY, ZX, X2 - Y2
        out_matrix = temp_list_out.extend([xyz_cs[0] * xyz_cs[1],
                                           xyz_cs[2] * xyz_cs[1], 1,
                                           xyz_cs[2] * xyz_cs[0], 1])
    # Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2
    out_matrix = np.array(temp_list_out)

    if manufacturer is not None:
        manufacturer = manufacturer.upper()
    if manufacturer == 'SIEMENS':
        out_matrix = _reorder_to_siemens(out_matrix, orders)
    elif manufacturer == 'GE':
        out_matrix = _reorder_to_ge(out_matrix, orders)
    elif manufacturer == 'PHILIPS':
        logger.warning("Philips shim CS not implemented yet")
    else:
        # Do not reorder if the manufacturer is not specified
        pass

    # None: Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2
    # GE: x, y, z, xy, zy, zx, X2 - Y2, z2
    # Siemens: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    return out_matrix
