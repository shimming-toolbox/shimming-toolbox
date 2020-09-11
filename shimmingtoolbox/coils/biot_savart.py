import numpy as np

# Code ported and refactored from Jason Stockmann and Fa-Hsuan Lin, "Magnetic field by Biot-Savart's Law" http://maki.bme.ntu.edu.tw/?page_id=333

# TODO: test against matlab and analytics, confirm this works
# TODO: docstrings

mu0 = 1

def _loop_segments(center, normal, radius, segment_num):
    segments = unit_circle = np.zeros((2, 3, segment_num))

    theta = np.linspace(0, 2 * np.pi, num=segment_num+1, endpoint=True).reshape((1, segment_num+1))
    unit_circle[0, :-1, :] = np.concatenate((np.cos(theta[:, :-1]), np.sin(theta[:, :-1])), axis=0) # Start points
    unit_circle[1, :-1, :] = np.concatenate((np.cos(theta[:, 1:]), np.sin(theta[:, 1:])), axis=0) # End points

    segments[:, :, :] = np.round(unit_circle * radius, decimals=5)
    segments = _rotate_z_to(normal) @ segments
    return segments + center.reshape((1, 3, 1))


def _rotate_z_to(normal):
    nhat = normal / np.linalg.norm(normal)

    rhat = np.cross((0, 0, 1), nhat)
    r_cross_mat = np.array([[0, -rhat[2], rhat[1]],
                     [rhat[2], 0, -rhat[0]],
                     [-rhat[1], rhat[0], 0]])
    cos = np.dot((0, 0, 1), nhat)
    sin = np.linalg.norm(rhat)
    if cos == 0:
        return np.identity(3) * sin
    R = np.identity(3) + r_cross_mat + r_cross_mat @ r_cross_mat * (1 - cos)/sin**2
    
    return R

def _integral(l, dl, r):
    rp = r - l
    B_per_I = mu0 / 4 * np.pi * np.cross(dl, rp) / np.linalg.norm(rp)**3
    return B_per_I[2]
    
def biot_savart(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n):
    ranges = []
    for i in range(3):
        ranges.append(np.linspace(fov_min[i], fov_max[i], num=fov_n[i]))
    x, y, z = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij', sparse=True)
    

    channels = len(centers)
    profiles = np.zeros((x.size, y.size, z.size, channels))
    for ch in range(channels):
        segments = _loop_segments(np.asarray(centers[ch]), np.asarray(normals[ch]), radii[ch], segment_numbers[ch])
        for segment in np.split(segments, segment_numbers[ch], axis=2):
            l = np.average(segment, axis=0).reshape(3)
            dl = (segment[1] - segment[0]).reshape(3)
            for i in range(x.size):
                for j in range(y.size):
                    for k in range(z.size):
                        profiles[i, j, k, ch] += _integral(l, dl, np.asarray([x[i, 0, 0], y[0, j, 0], z[0, 0, k]]))
    return profiles

    


print(biot_savart([(0, 0, 2)], [(1, 0, 0)], [2], [4], (0, 0, 0), (2, 2, 2), (3, 3, 3)))
