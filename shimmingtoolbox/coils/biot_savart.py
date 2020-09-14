#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code ported and refactored from Jason Stockmann and Fa-Hsuan Lin, "Magnetic field by Biot-Savart's Law" http://maki.bme.ntu.edu.tw/?page_id=333

import numpy as np

mu0 = 1.256637e-6 # [H/m]
H_gyromagnetic_ratio = 4.258e+7 # [Hz/T]

def biot_savart(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n):
    """Creates coil profiles for arbitrary loops, for use in multichannel shim examples that do not match spherical harmonics
    Args:
        centers (list of (3,) array-like floats): List of center points for each loop in mm
        normals (list of (3,) array-like 3D floats): List of 3D normal vectors for each loop in mm
        radii (list of floats): List of radii for each loop in mm
        segment_numbers (list of ints): List of the number of segments for each loop approximation
        fov_min ((3,) array-like of floats): Low corner of coil profile field of view (x, y, z)
        fov_max ((3,) array-like of floats): Inclusive high corner of coil profile field of view (x, y, z)
        fov_n ((3,) array-like of ints): Number of points for each dimension (x, y, z)

    Returns:
        numpy.ndarray: (|X|, |Y|, |Z|, |centers|) coil profiles of magnetic field z-component -- (X, Y, Z, Channel)

    """
    ranges = []
    for i in range(3):
        ranges.append(np.linspace(fov_min[i], fov_max[i], num=fov_n[i]))
    x, y, z = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij', sparse=True)
    

    channels = len(centers)
    profiles = np.zeros((x.size, y.size, z.size, channels))
    for ch in range(channels):
        segments = _loop_segments(np.asarray(centers[ch]), np.asarray(normals[ch]), radii[ch], segment_numbers[ch])
        print(f"Channel {ch}")
        n = 0
        for segment in np.split(segments, segment_numbers[ch], axis=2):
            n += 1
            print(f"Segment {n}")
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

    return profiles

def _loop_segments(center, normal, radius, segment_num):
    """Creates loop segments for loop approximation, given loop details
    Args:
        center ((3,) numpy.ndarray): 3D center points loop in arbitrary units
        normal ((3,) numpy.ndarray): Normal vector to loop in arbitrary units
        radius (float): Loop radius in arbitrary units
        segment_num (int): Number of segments for loop approximation 

    Returns:
        numpy.ndarray: (2, 3, segment_num) array of segments (segment start [0] or end [1]; x, y, z ; segment number)
    """
    segments = unit_circle = np.zeros((2, 3, segment_num))

    theta = np.linspace(0, 2 * np.pi, num=segment_num+1, endpoint=True).reshape((1, segment_num+1))
    unit_circle[0, :-1, :] = np.concatenate((np.cos(theta[:, :-1]), np.sin(theta[:, :-1])), axis=0) # Start points
    unit_circle[1, :-1, :] = np.concatenate((np.cos(theta[:, 1:]), np.sin(theta[:, 1:])), axis=0) # End points

    segments[:, :, :] = np.round(unit_circle * radius, decimals=9)
    segments = _rotate_z_to(normal) @ segments
    return segments + center.reshape((1, 3, 1))


def _rotate_z_to(target):
    """Creates 3D rotation matrix that sends (0, 0, 1) to target vector
    Args:
        target ((3,) numpy.ndarray): Target vector to rotate (0, 0, 1) to

    Returns:
        numpy.ndarray: (3, 3) 3D rotation matrix sending (0, 0, 1) to target vector
    """
    vhat = target / np.linalg.norm(target)
    rhat = np.cross((0, 0, 1), vhat)

    r_cross_mat = np.array([[0, -rhat[2], rhat[1]],
                     [rhat[2], 0, -rhat[0]],
                     [-rhat[1], rhat[0], 0]])

    cs = np.dot((0, 0, 1), vhat) # cosine
    sn = np.linalg.norm(rhat) # sine

    if sn == 0: # Target is parallel to z-axis
        return np.array([[1, 0, 0],
                        [0, cs, 0],
                        [0, 0, cs]]) # Return rotation around x (180 or 0 degree)

    else:
        return np.identity(3) + r_cross_mat + r_cross_mat @ r_cross_mat * (1 - cs)/sn**2

def _z_field(l, dl, r):
    """Calculate z-field at point r from line segment centered at l with length dl
    Args:
        l ((3,) numpy.ndarray): Line segment center in m
        dl ((3,) numpy.ndarray): Line segment vector in m
        r ((3,) numpy.ndarray): Target point in m

    Returns:
        float: z-component of magnetic field at r in T/A
    """
    l, dl, r = l / 1000, dl / 1000, r / 1000 # Convert mm to m
    rp = r - l
    rp_norm = np.linalg.norm(rp)
    if rp_norm == 0:
        return np.nan
    B_per_I = mu0 / (4 * np.pi) * np.cross(dl, rp) / rp_norm ** 3
    return B_per_I[2] # [T/A]