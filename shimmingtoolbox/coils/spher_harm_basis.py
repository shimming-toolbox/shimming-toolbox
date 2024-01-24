#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import numpy as np

from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics

logger = logging.getLogger(__name__)

GYROMAGNETIC_RATIO = 42.5774785178325552  # [MHz/T] or equivalently [Hz/uT]
SHIM_CS = {'SIEMENS': 'LAI',
           'GE': 'LPI',
           'PHILIPS': 'RPI'}


def siemens_basis(x, y, z, orders=(1, 2), shim_cs=SHIM_CS['SIEMENS']):
    """
    The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. *Following Siemens convention*,
    ``basis`` is then:

        - Rescaled to Hz/unit-shim, where "unit-shim" refers to the measure displayed in the Adjustments card of the
          Syngo console UI, namely:

            - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
            - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)
            - 1 micro-T/m^3 for 3rd order terms (= 0.000000042576 Hz/mm^3)

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
    flip = get_flip_matrix(shim_cs, manufacturer='SIEMENS', orders=(1,))
    spher_harm = scaled_spher_harm(x * flip[0], y * flip[1], z * flip[2], orders)

    # Reorder according to siemens convention: X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
    reordered_spher = reorder_to_manufacturer(spher_harm, manufacturer='SIEMENS')

    # Convert back to an array
    output = convert_spher_harm_to_array(reordered_spher)

    return output


def ge_basis(x, y, z, orders=(1, 2), shim_cs=SHIM_CS['GE']):
    """
    The function first wraps ``shimmingtoolbox.coils.spher_harm_basis.scaled_spher_harm`` to generate 1st and 2nd
    order spherical harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``.
    *Following GE convention*, ``basis`` is then:

        - Reordered along the 4th dimension as *x, y, z, xy, zy, zx, X2-Y2, z2*

        - Rescaled:

            - 1 XXXXX for *X,Y,Z* gradients (= Hz/mm)
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
    flip = get_flip_matrix(shim_cs, manufacturer='GE', orders=(1,))
    spher_harm = scaled_spher_harm(x * flip[0], y * flip[1], z * flip[2], orders)

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

    # Reorder according to GE shim convention: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY
    reordered_spher = reorder_to_manufacturer(spher_harm, manufacturer='GE')

    scaled = {}
    for order in orders:
        if order == 0:
            scaled[0] = reordered_spher[0]
        elif order == 1:
            # Rescale to unit-shims that are XXXXX
            # They are currently in uT/m
            # Todo: This seems to be the appropriate scaling factor, we need to verify the units
            scaled[1] = reordered_spher[1] / 10
        elif order == 2:

            def _reorder_shim_to_scaling(coefs):
                # Reorder 2nd order terms
                # 1. * Z2, ZX, ZY, X2 - Y2, XY * (in line with GE shims)
                # 2. * xy, zy, zx, X2 - Y2, z2 * (scaling matrix)
                return coefs[..., [4, 2, 1, 3, 0]]

            def _reorder_scaling_to_shim(coefs):
                # Reorder 2nd order terms
                # 1. * xy, zy, zx, X2 - Y2, z2 * (scaling matrix)
                # 2. * Z2, ZX, ZY, X2 - Y2, XY * (in line with GE shims)
                return coefs[..., [4, 2, 1, 3, 0]]

            # Scale
            # Hz/cm2/A, -> uT/m2/A = order2_to_order2 * 1e6 * (100 ** 2) / (GYROMAGNETIC_RATIO * 1e6)
            # = order2_to_order2 * (100 ** 2) / GYROMAGNETIC_RATIO
            orders_to_order2_uT = order2_to_order2 * (100 ** 2) / GYROMAGNETIC_RATIO

            # Reorder 2nd order terms to the scaling matrix order
            # *xy, zy, zx, X2 - Y2, z2 * (scaling matrix)
            reordered_spher[2] = _reorder_shim_to_scaling(reordered_spher[2])

            scaled[2] = np.zeros_like(reordered_spher[2])
            for i_channel in range(reordered_spher[order].shape[-1]):
                # Since reordered_spher contains the values of 1uT/m^2 in Hz/mm^2. We simply multiply by the amount of
                # uT/m^2 / A
                # This gives us a value in Hz/mm^2 / A which we need to modify to Hz/mm^2 / mA
                scaled[2][..., i_channel] = np.matmul(reordered_spher[2],
                                                      orders_to_order2_uT[i_channel, :]) / 1000
                # Todo: We need a /2 between expected zx, zy, xy results and calculated results
                if i_channel in [0, 1, 2]:
                    scaled[2][..., i_channel] /= 2

            # Reorder 2nd order terms to the shim order
            # 2. * Z2, ZX, ZY, X2 - Y2, XY * (in line with GE shims)
            scaled[2] = _reorder_scaling_to_shim(scaled[2])

        else:
            logger.warning(f"Scaling spherical harmonics of order {order} not implemented for GE")
            scaled[order] = reordered_spher[order]

    # Convert back to an array
    output = convert_spher_harm_to_array(scaled)

    return output


def philips_basis(x, y, z, orders=(1, 2), shim_cs=SHIM_CS['PHILIPS']):
    """
    The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. *Following Philips convention*,
    ``basis`` is then:

        - Rescaled to Hz/unit-shim, where "unit-shim" refers to:

            - 1 milli-T/m for *X,Y,Z* gradients (= 42.576 Hz/mm)
            - 1 milli-T/m^2 for 2nd order terms (= 0.042576 Hz/mm^2)

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

    Note:
        Philips coordinate system has its x in the AP direction and y axis in the RL direction. Therefore, channel 0 (x)
        changes along axis 1 and channel 1 (y) changes along axis 0.
    """
    # Check inputs
    _check_basis_inputs(x, y, z, orders)

    # Create spherical harmonics from first to second order
    # Philips' y and x axis are flipped (x is AP, y is RL)
    flip = get_flip_matrix(shim_cs, manufacturer='Philips', orders=(1,))
    spher_harm = scaled_spher_harm(y * flip[0], x * flip[1], z * flip[2], orders)

    # Reorder according to philips convention: X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
    reordered_spher = reorder_to_manufacturer(spher_harm, manufacturer='PHILIPS')

    # Scale according to Philips convention
    # milli-T/m for order 1, milli-T/m^2 for order 2
    # uT/m * 1e3 = mT/m, uT/m^2 * 1e3 = mT/m^2
    for order in orders:
        if order == 0:
            continue

        reordered_spher[order] *= 1e3

        if order == 2:
            # Todo: We need a /2 between expected zx, zy results and calculated results (Similar to GE but not with XY)
            reordered_spher[order][..., 1] /= 2
            reordered_spher[order][..., 2] /= 2
        else:
            logger.warning(f"Scaling spherical harmonics of order {order} not implemented for Philips")

    output = convert_spher_harm_to_array(reordered_spher)

    return output


def scaled_spher_harm(x, y, z, orders=(1, 2)):
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

    Returns:
        dict: dictionary of the basis set of spherical harmonics scaled
    """
    # Check inputs
    _check_basis_inputs(x, y, z, orders)

    # Create spherical harmonics from first to second order
    spher_harm = spherical_harmonics(orders, x, y, z)
    # 1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics)

    # Convert to a dictionary of 3D arrays, where each key is the order of the spherical harmonic
    spher_harm_dict = convert_spher_harm_to_dict(spher_harm, orders)

    # scale according to
    # - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
    # - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)
    # - 1 micro-T/m^3 for 3rd order terms (= 0.000000042576 Hz/mm^3)
    scaling_factors = _get_scaling_factors(orders)
    scaling_factors_dict = convert_spher_harm_to_dict(scaling_factors, orders)

    scaled = {}
    for order in orders:
        scaled[order] = scaling_factors_dict[order] * spher_harm_dict[order]

    # 1 uT/m, 1 uT/m2, 1uT/mm3 in Hz/mm, Hz/mm2, Hz/mm3 respectively
    return scaled


def convert_spher_harm_to_dict(spher_harm, orders):
    """ Convert an array of spherical harmonics to a dictionary of 3D/4d arrays, where each key is the order of the

    Args:
        spher_harm (np.ndarray): Array of spherical harmonics
        orders (tuple): Tuple containing the orders of the spherical harmonics in the array, sorted in ascending order

    Returns:
        dict: Dictionary of 3D arrays, where each key is the order of the spherical harmonic
    """

    spher_harm_dict = {}
    i_ch = 0
    for order in orders:
        spher_harm_dict[order] = spher_harm[..., i_ch:i_ch + channels_per_order(order)]
        i_ch += channels_per_order(order)

    return spher_harm_dict


def convert_spher_harm_to_array(spher_harm_dict):
    spher_harm = []
    for order in sorted(spher_harm_dict.keys()):
        spher_harm.append(spher_harm_dict[order])

    spher_harm = np.concatenate(spher_harm, axis=-1)

    return spher_harm


def reorder_to_manufacturer(spher_harm, manufacturer):
    """
    Reorder 1st - 2nd - 3rd order coefficients, if specified. From

    Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, y(x2 - y2), xyz, yz2, z3, xz^2, z(x2 - y2), x(x2 - y2)
    (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, z3,  xz^2, yz2, z(x2 - y2) (in line with Siemens shims) or

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with GE shims) or

    X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with Philips shims)

    Args:
        spher_harm (dict): 3D array of spherical harmonics coefficients with key corresponding to the order
        manufacturer (str): Manufacturer of the scanner

    Returns:
        dict: Coefficients ordered following the manufacturer's convention
    """

    def _reorder_order0(sph, manuf):
        if sph.shape[-1] != 1:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 1")
        return sph[..., [0]]

    def _reorder_order1(sph, manuf):
        if sph.shape[-1] != 3:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 3")
        if manuf in ['SIEMENS', 'GE', 'PHILIPS']:
            return sph[..., [2, 0, 1]]
        else:
            logger.warning(f"1st order spherical harmonics not implemented for: {manuf}")
            return sph

    def _reorder_order2(sph, manuf):
        if sph.shape[-1] != 5:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 5")

        if manuf in ['SIEMENS', 'PHILIPS', 'GE']:
            return sph[..., [2, 3, 1, 4, 0]]
        else:
            logger.warning(f"2nd order spherical harmonics not implemented for: {manuf}")
            return sph

    def _reorder_order3(sph, manuf):
        if sph.shape[-1] != 7:
            raise ValueError("Input arrays should have 4th dimension's shape equal to 7")
        if manufacturer == 'SIEMENS':
            return sph[..., [3, 4, 2, 5]]
        else:
            logger.warning(f"3rd order spherical harmonics not implemented for: {manuf}")
            return sph

    reorder = {0: _reorder_order0,
               1: _reorder_order1,
               2: _reorder_order2,
               3: _reorder_order3}

    reordered = {}
    for order in spher_harm.keys():
        if order not in reorder.keys():
            logger.warning(f"Ordering for order {order} spherical harmonics not implemented")
        reordered[order] = reorder[order](spher_harm[order], manuf=manufacturer)

    return reordered


def _get_scaling_factors(orders):
    """
    Get scaling factors for the 1st/2nd/3rd order spherical harmonic
    fields for rescaling them to 1 uT/unit-shim in units of Hz/mm:

    Gx, Gy, and Gz should yield 1 micro-T of field shift per metre: equivalently, 0.042576 Hz/mm

    2nd order terms should yield 1 micro-T of field shift per metre-squared: equivalently, 0.000042576 Hz/mm^2

    3rd order terms should yield 1 micro-T of field shift per metre-cubed: equivalently, 0.000000042576 Hz/mm^3

    Gist: given the stated nominal values, we can pick several arbitrary reference positions around the
    origin/isocenter at which we know what the field *should* be, and use that to calculate the appropriate scaling
    factor.

    Returns:
         numpy.ndarray:  1D vector of ``scaling_factors``
    """

    [x_iso, y_iso, z_iso] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                        indexing='xy')
    sh = spherical_harmonics(orders, x_iso, y_iso, z_iso)

    n_channels = np.array([channels_per_order(order) for order in orders]).sum()
    scaling_factors = np.zeros(n_channels)

    # indices of reference positions for normalization:
    # needed for order 1
    i_x1 = np.nonzero((x_iso == 1) & (y_iso == 0) & (z_iso == 0))
    i_y1 = np.nonzero((x_iso == 0) & (y_iso == 1) & (z_iso == 0))
    i_z1 = np.nonzero((x_iso == 0) & (y_iso == 0) & (z_iso == 1))
    # needed for order 2
    i_x1z1 = np.nonzero((x_iso == 1) & (y_iso == 0) & (z_iso == 1))
    i_y1z1 = np.nonzero((x_iso == 0) & (y_iso == 1) & (z_iso == 1))
    i_x1y1 = np.nonzero((x_iso == 1) & (y_iso == 1) & (z_iso == 0))
    # needed for order 3
    i_x1y1z1 = np.nonzero((x_iso == 1) & (y_iso == 1) & (z_iso == 1))
    # needed for order 4
    # i_x2y1 = np.nonzero((x_iso == 2) & (y_iso == 1) & (z_iso == 0))

    # order the reference indices like the sh field terms
    # TODO: Find out the polarity of the terms
    # 1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, YZ2, Z3, XZ^2, Z(X2 - Y2), X(X2 - Y2)
    # (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics)
    iref = {0: [i_x1],
            1: [i_y1, i_z1, i_x1],
            2: [i_x1y1, i_y1z1, i_z1, i_x1z1, i_x1],
            3: [i_y1, i_x1y1z1, i_y1, i_z1, i_x1, i_x1z1, i_x1],
            # 4: [i_x2y1, i_y1z1, i_x1y1, i_y1z1, i_x1, i_x1z1, i_x1, i_x1z1, i_x1]
            }

    # distance from iso/origin to adopted reference point[units: mm]
    r = {0: [1],
         1: [1, 1, 1],
         2: [np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1],
         3: [1, np.sqrt(3), 1, 1, 1, np.sqrt(2), 1],
         # 4: [np.sqrt(5), np.sqrt(2), np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1]
         }

    i_ch = 0
    for order in orders:

        if r.get(order) is None or iref.get(order) is None:
            raise NotImplementedError("Order must be between 0 and 3")

        for i in range(channels_per_order(order)):
            field = sh[:, :, :, i_ch]
            if order != 0:
                scaling_factors[i_ch] = (GYROMAGNETIC_RATIO * ((r[order][i] * 0.001) ** order) /
                                         field[iref[order][i]][0])
            else:
                scaling_factors[i_ch] = -1 / field[iref[order][i]][0]

            i_ch += 1

    return scaling_factors


def _check_basis_inputs(x, y, z, orders):
    # Check inputs
    if not (x.ndim == y.ndim == z.ndim == 3):
        raise RuntimeError("Input arrays X, Y, and Z must be 3d")

    if not (x.shape == y.shape == z.shape):
        raise RuntimeError("Input arrays X, Y, and Z must be identically sized")

    if max(orders) >= 3:
        raise NotImplementedError("Spherical harmonics not implemented for order 4 and up")


def channels_per_order(order, manufacturer=None):
    """
    Return the number of channels per order for the specified manufacturer

    Args:
        order (int): Order of the spherical harmonics
        manufacturer (str): Manufacturer of the scanner.

    Returns:

    """
    if manufacturer == 'Siemens' and order == 3:
        return 4
    return 2 * order + 1


def get_flip_matrix(shim_cs='RAS', manufacturer=None, orders=None):
    """
    Return a matrix to flip the spherical harmonics basis set from RAS to the desired coordinate system.

    Args:
        shim_cs (str): Coordinate system of the shim basis set. Default is RAS.
        orders (list): List of orders of the spherical harmonics. Default to None (all orders)
        manufacturer (str): Manufacturer of the scanner. The flipping matrix is different for each manufacturer.
                            If None is selected, it will output according to
                            ``shimmingtoolbox.coils.spherical_harmonics``. Possible values: SIEMENS, GE, PHILIPS.

    Returns:
        numpy.ndarray: Matrix (len: 8) to flip the spherical harmonics basis set from ras to the desired coordinate
                       system. Output is a 1D vector of ``flip_matrix`` for the following:
                       Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, YZ2, Z3, XZ^2, Z(X2 - Y2), X(X2 - Y2).
                       If xyz is True, output X, Y, Z only in this order.
    """
    if orders is None:
        orders = [1, 2, 3]

    xyz_cs = [1, 1, 1]

    shim_cs = shim_cs.upper()
    if (len(shim_cs) != 3) or \
            (shim_cs[0] not in ['R', 'L']) or (shim_cs[1] not in ['A', 'P']) or (shim_cs[2] not in ['S', 'I']):
        raise ValueError(f"Unknown coordinate system: {shim_cs}")

    if shim_cs[0] == 'L':
        xyz_cs[0] = -1
    if shim_cs[1] == 'P':
        xyz_cs[1] = -1
    if shim_cs[2] == 'I':
        xyz_cs[2] = -1

    # Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, YZ2, Z3, XZ^2, Z(X2 - Y2), X(X2 - Y2)
    out_dict = {}
    for order in orders:
        if order == 1:
            out_dict[1] = np.array([xyz_cs[1], xyz_cs[2], xyz_cs[0]])
        if order == 2:
            out_dict[2] = np.array([xyz_cs[0] * xyz_cs[1], xyz_cs[2] * xyz_cs[1], 1, xyz_cs[2] * xyz_cs[0], 1])
        if order == 3:
            # TODO: Verify
            out_dict[3] = np.array([xyz_cs[1], xyz_cs[0] * xyz_cs[1] * xyz_cs[2], xyz_cs[1], xyz_cs[2], xyz_cs[0],
                                    xyz_cs[2], xyz_cs[0]])

    if manufacturer is not None:
        manufacturer = manufacturer.upper()

    if manufacturer in ['SIEMENS', 'GE', 'PHILIPS']:
        out_dict = reorder_to_manufacturer(out_dict, manufacturer)
    else:
        # Do not reorder if the manufacturer is not specified
        logger.warning(f"Flip matrix not implemented for manufacturer: {manufacturer}")
        pass

    out_list = []
    for i_order in sorted(orders):
        out_list += out_dict[i_order].tolist()

    # None: Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2, Y(X2 - Y2), XYZ, YZ2, Z3, XZ^2, Z(X2 - Y2), X(X2 - Y2)
    # GE: x, y, z, xy, zy, zx, X2 - Y2, z2, 3rd order not implemented
    # Siemens: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, Z3,  XZ^2, YZ2, Z(X2 - Y2)
    # Philips: X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY, 3rd order not implemented
    return out_list
