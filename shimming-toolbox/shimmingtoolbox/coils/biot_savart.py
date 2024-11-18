#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code ported and refactored from Jason Stockmann and Fa-Hsuan Lin, "Magnetic field by Biot-Savart's Law"
# http://maki.bme.ntu.edu.tw/?page_id=333

import numpy as np

MU0 = 1.256637e-6  # [H/m]
H_GYROMAGNETIC_RATIO = 42.577478518e+6  # [Hz/T]


def biot_savart(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n):
    """
    Creates coil profiles for arbitrary loops, for use in multichannel shim examples that do not match spherical
    harmonics
    Args:
        centers (list): List of 3D float center points for each loop in mm
        normals (list): List of 3D float normal vectors for each loop in mm
        radii (list): List of float radii for each loop in mm
        segment_numbers (list): List of integer number of segments for each loop approximation
        fov_min (tuple): Low 3D float corner of coil profile field of view (x, y, z) in mm
        fov_max (tuple): Inclusive high 3D float corner of coil profile field of view (x, y, z) in mm
        fov_n (tuple): Integer number of points for each dimension (x, y, z) in mm

    Returns:
        numpy.ndarray: (X, Y, Z, centers) coil profiles of magnetic field z-component in Hz/A -- (X, Y, Z, Channel)

    """
    ranges = []
    for i in range(3):
        ranges.append(np.linspace(fov_min[i], fov_max[i], num=fov_n[i]))
    x, y, z = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij', sparse=True)

    channels = len(centers)
    profiles = np.zeros((x.size, y.size, z.size, channels))
    for ch in range(channels):
        segments = _loop_segments(np.asarray(centers[ch]), np.asarray(normals[ch]), radii[ch], segment_numbers[ch])
        n = 0
        for segment in np.split(segments, segment_numbers[ch], axis=2):
            n += 1
            l = np.average(segment, axis=0).reshape(3)
            dl = (segment[1] - segment[0]).reshape(3)
            for i in range(x.size):
                for j in range(y.size):
                    for k in range(z.size):
                        bz = _z_field(l, dl, np.asarray([x[i, 0, 0], y[0, j, 0], z[0, 0, k]]))
                        if np.isnan(bz):
                            profiles[i, j, k, ch] = bz
                        if not np.isnan(profiles[i, j, k, ch]):
                            profiles[i, j, k, ch] += bz

    return profiles * H_GYROMAGNETIC_RATIO  # [Hz/A]


def _loop_segments(center, normal, radius, segment_num):
    """Creates loop segments for loop approximation, given loop details
    Args:
        center (numpy.ndarray): 3D center point of loop in arbitrary units
        normal (numpy.ndarray): 3D normal vector to loop in arbitrary units
        radius (float): Loop radius in arbitrary units
        segment_num (int): Number of segments for loop approximation

    Returns:
        numpy.ndarray: (2, 3, segment_num) array of segments (segment start [0] or end [1]; x, y, z ; segment number)
    """
    segments = unit_circle = np.zeros((2, 3, segment_num))

    theta = np.linspace(0, 2 * np.pi, num=segment_num+1, endpoint=True).reshape((1, segment_num+1))
    unit_circle[0, :-1, :] = np.concatenate((np.cos(theta[:, :-1]), np.sin(theta[:, :-1])), axis=0)  # Start points
    unit_circle[1, :-1, :] = np.concatenate((np.cos(theta[:, 1:]), np.sin(theta[:, 1:])), axis=0)  # End points

    segments[:, :, :] = np.round(unit_circle * radius, decimals=9)
    segments = _rotate_z_to(normal) @ segments
    return segments + center.reshape((1, 3, 1))


def _rotate_z_to(target):
    """Creates 3D rotation matrix that sends (0, 0, 1) to target vector
    Args:
        target (numpy.ndarray): 3D float target vector to rotate (0, 0, 1) to

    Returns:
        numpy.ndarray: (3, 3) 3D rotation matrix sending (0, 0, 1) to target vector
    """
    vhat = target / np.linalg.norm(target)
    rhat = np.cross((0, 0, 1), vhat)

    r_cross_mat = np.array([[0, -rhat[2], rhat[1]],
                            [rhat[2], 0, -rhat[0]],
                            [-rhat[1], rhat[0], 0]])

    cs = np.dot((0, 0, 1), vhat)  # cosine
    sn = np.linalg.norm(rhat)  # sine

    if sn == 0:  # Target is parallel to z-axis
        return np.array([[1, 0, 0],
                        [0, cs, 0],
                        [0, 0, cs]])  # Return rotation around x (180 or 0 degree)

    else:
        return np.identity(3) + r_cross_mat + r_cross_mat @ r_cross_mat * (1 - cs)/sn**2


def _z_field(l, dl, r):
    """Calculate z-field at point r from line segment centered at l with length dl
    Args:
        l (numpy.ndarray): Line segment center in m
        dl (numpy.ndarray): Line segment vector in m
        r (numpy.ndarray): Target point in m

    Returns:
        float: z-component of magnetic field at r in T/A
    """
    l, dl, r = l / 1000, dl / 1000, r / 1000  # Convert mm to m
    rp = r - l
    rp_norm = np.linalg.norm(rp)
    if rp_norm == 0:
        return np.nan
    b_per_i = MU0 / (4 * np.pi) * np.cross(dl, rp) / rp_norm ** 3
    return b_per_i[2]  # [T/A]


def generate_coil_bfield(wire, xyz, grid_size):
    """Generates Bz field in the FOV

    Args:
        wire (list): 1D list of n_segments dictionaries with start and stop point of the segment
        xyz (np.array): 2D array shape (n_points, 3) where n_points is the number of points in the whole FOV. Represents
                            the (x, y, z) coordinates in mm of each point in the FOV
        grid_size (tuple): Shape of the FOV

    Returns:
        numpy.ndarray: Bz field shaped back to grid_size
    """
    n_positions = xyz.shape[0]
    fz = np.zeros((n_positions, 1))
    n_segments = len(wire)

    def integral(p, q, a, b, c):
        term1 = q * (2 * (2 * a + b) / (4 * a * c - b**2) / np.sqrt(a + b + c) - 2 * b / (4 * a * c - b**2) /
                     np.sqrt(c))
        term2 = p * (2 * (b + 2 * c) / (b**2 - 4 * a * c) / np.sqrt(a + b + c) - 4 * c / (b**2 - 4 * a * c) /
                     np.sqrt(c))
        output = term1 + term2
        return output

    for i_segment in range(n_segments):
        if 'weight' in wire[0]:
            w = wire[i_segment]['weight']
        else:
            w = 1.0

        a = np.tile(np.linalg.norm(wire[i_segment]['start'] - wire[i_segment]['stop'])**2, (n_positions, 1))
        b = 2 * np.sum(np.tile(wire[i_segment]['stop'] - wire[i_segment]['start'], (n_positions, 1)) *
                       (np.tile(wire[i_segment]['start'], (n_positions, 1)) - xyz), axis=1, keepdims=True)
        c = np.sum((np.tile(wire[i_segment]['start'], (n_positions, 1)) - xyz)**2, axis=1, keepdims=True)

        s1 = np.tile(wire[i_segment]['start'], (n_positions, 1))
        s2 = np.tile(wire[i_segment]['stop'], (n_positions, 1))

        pz = (s2[:,0] - s1[:,0]) * (s2[:,1] - s1[:,1]) - (s2[:,1] - s1[:,1]) * (s2[:,0] - s1[:,0])
        qz = (s2[:,0] - s1[:,0]) * (s1[:,1] - xyz[:,1]) - (s2[:,1] - s1[:,1]) * (s1[:,0] - xyz[:,0])
        pz = np.reshape(pz, (n_positions, 1))
        qz = np.reshape(qz, (n_positions, 1))

        fz += integral(pz, qz, a, b, c) * w

    bz = np.reshape(fz, grid_size, order='F') / 1e4

    return bz
