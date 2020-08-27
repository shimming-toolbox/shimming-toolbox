#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
from shimmingtoolbox.coils.spherical_harmonics import spherical_harmonics


def siemens_basis(x, y, z):
    """
    The function first wraps ``shimmingtoolbox.coils.spherical_harmonics`` to generate 1st and 2nd order spherical
    harmonic ``basis`` fields at the grid positions given by arrays ``X,Y,Z``. *Following Siemens convention*, ``basis`` is
    then:

        - Reordered along the 4th dimension as *X, Y, Z, Z2, ZX, ZY, X2-Y2, XY*

        - Rescaled to Hz/unit-shim, where "unit-shim" refers to the measure displayed in the Adjustments card of the
          Syngo console UI, namely:

            - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
            - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)

    The returned ``basis`` is thereby in the form of ideal "shim reference maps", ready for optimization.

    Args:
        orders (numpy.ndarray): **not yet implemented** Degrees of the desired terms in the series expansion, specified
                                as a vector of non-negative integers (``[0:1:n]`` yields harmonics up to n-th order)
        x (numpy.ndarray): 3-D arrays of grid coordinates, "Right->Left" grid coordinates in the patient coordinate
                           system (i.e. DICOM reference, units of mm)
        y (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Anterior->Posterior" grid coordinates in
                           the patient coordinate system (i.e. DICOM reference, units of mm)

        z (numpy.ndarray): 3-D arrays of grid coordinates (same shape as x). "Inferior->Superior" grid coordinates in
                           the patient coordinate system (i.e. DICOM reference, units of mm)

    Returns:
        numpy.ndarray: 4-D array of spherical harmonic basis fields

    NOTES:
        For now, ``orders`` is, in fact, ignored: fixed as [1:2]—which is suitable for the Prisma (presumably other
        Siemens systems as well) however, the 3rd-order shims of the Terra should ultimately be accommodated too.
        (Requires checking the Adjustments/Shim card to see what the corresponding terms and values actually are). So,
        for now, ``basis`` will always be returned with 8 terms along the 4th dim.
    """

    # Local functions
    def reorder_to_siemens(spher_harm):
        """
        Reorder 1st - 2nd order basis terms along 4th dim. From
        1. Y, Z, X, XY, ZY, Z2, ZX, X2 - Y2 (output by shimmingtoolbox.coils.spherical_harmonics.spherical_harmonics), to
        2. X, Y, Z, Z2, ZX, ZY, X2 - Y2, XY (in line with Siemens shims)
        """

        assert spher_harm.shape[3] == 8
        reordered = np.zeros(spher_harm.shape)
        reordered[:, :, :, 0] = spher_harm[:, :, :, 2]
        reordered[:, :, :, 1] = spher_harm[:, :, :, 0]
        reordered[:, :, :, 2] = spher_harm[:, :, :, 1]
        reordered[:, :, :, 3] = spher_harm[:, :, :, 5]
        reordered[:, :, :, 4] = spher_harm[:, :, :, 6]
        reordered[:, :, :, 5] = spher_harm[:, :, :, 4]
        reordered[:, :, :, 6] = spher_harm[:, :, :, 7]
        reordered[:, :, :, 7] = spher_harm[:, :, :, 3]

        return reordered

    def get_scaling_factors():
        """
        Returns a vector of ``scaling_factors`` to apply to the(Siemens-reordered) 1st + 2nd order spherical harmonic fields
        for rescaling field terms as "shim reference maps" in units of Hz/unit-shim:

        Gx, Gy, and Gz should yield 1 micro-T of field shift per metre: equivalently, 0.042576 Hz/mm

        2nd order terms should yield 1 micro-T of field shift per metre-squared: equivalently, 0.000042576 Hz/mm^2

        Gist: given the stated nominal values, we can pick several arbitrary reference positions around the origin/isocenter
        at which we know what the field *should* be, and use that to calculate the appropriate scaling factor.

        NOTE: The method has been worked out empirically and has only been tested for 2 Siemens Prisma systems.
        E.g.re: Y, Z, ZX, and XY terms, their polarity had to be flipped relative to the form given by
        shimmingtoolbox.coils.spherical_harmonics. To adapt the code to other systems, arbitrary(?) changes along these
        lines will likely be needed.
        """

        [x_iso, y_iso, z_iso] = np.meshgrid(np.array(range(-1, 2)), np.array(range(-1, 2)), np.array(range(-1, 2)),
                                            indexing='xy')
        orders = np.array(range(1, 3))
        sh = spherical_harmonics(orders, x_iso, y_iso, z_iso)
        sh = reorder_to_siemens(sh)

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

        # invert polarity
        # Y, Z, ZX, and XY terms only(determined empirically)
        sh[:, :, :, [1, 2, 4, 7]] = -sh[:, :, :, [1, 2, 4, 7]]

        # scaling:
        orders = [1, 1, 1, 2, 2, 2, 2, 2]

        for i_ch in range(0, n_channels):
            field = sh[:, :, :, i_ch]
            scaling_factors[i_ch] = 42.576 * ((r[i_ch] * 0.001) ** orders[i_ch]) / field[i_ref[i_ch]]

        return scaling_factors

    # Check inputs
    if not (x.ndim == y.ndim == z.ndim == 3):
        raise RuntimeError("Input arrays X, Y, and Z must be 3d")

    if not (x.shape == y.shape == z.shape):
        raise RuntimeError("Input arrays X, Y, and Z must be identically sized")

    # Create spherical harmonics from first to second order
    orders = np.array(range(1, 3))
    spher_harm = spherical_harmonics(orders, x, y, z)

    # Reorder according to siemens convention: X, Y, Z, Z2, ZX, ZY, X2-Y2, XY
    reordered_spher = reorder_to_siemens(spher_harm)

    # scale according to
    # - 1 micro-T/m for *X,Y,Z* gradients (= 0.042576 Hz/mm)
    # - 1 micro-T/m^2 for 2nd order terms (= 0.000042576 Hz/mm^2)
    scaling_factors = get_scaling_factors()
    scaled = np.zeros_like(reordered_spher)
    for i_channel in range(0, spher_harm.shape[3]):
        scaled[:, :, :, i_channel] = scaling_factors[i_channel] * reordered_spher[:, :, :, i_channel]

    return scaled
