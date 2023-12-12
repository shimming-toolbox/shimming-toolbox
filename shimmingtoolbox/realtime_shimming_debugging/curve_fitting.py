import nibabel as nib
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.optimize as opt

from sklearn.linear_model import Lasso, Ridge
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.load_nifti import get_acquisition_times
from sklearn.preprocessing import PolynomialFeatures

# PATHS
FIELD_MAP_PATH_1 = './rt_dataset/dataset_1/rt_fm_dataset_1.nii.gz'
MAG_PATH_1 = './rt_dataset/dataset_1/rt_mag_dataset_1.nii.gz'
FNAME_JSON_1 = FIELD_MAP_PATH_1.rsplit('.nii', 1)[0] + '.json'

FIELD_MAP_PATH_2 = './rt_dataset/dataset_2/rt_fm_dataset_2.nii.gz'
MAG_PATH_2 = './rt_dataset/dataset_2/rt_mag_dataset_2.nii.gz'
FNAME_JSON_2 = FIELD_MAP_PATH_2.rsplit('.nii', 1)[0] + '.json'

MASK_PATH = '/Users/arnaud/Documents/MSC/UNF_data/acdc_216/centerline.nii.gz'
PMU_PATH = '/Users/arnaud/Documents/MSC/UNF_experiment/acdc_216/PMUresp_signal_acqs.resp'


def load_data(FIELD_MAP_PATH, MAG_PATH, FNAME_JSON, MASK_PATH, PMU_PATH):
    # Load data
    field_map_nib = nib.load(FIELD_MAP_PATH)
    field_map_data = field_map_nib.get_fdata()[..., 0, :] # remove extra dimension with size 1
    mag = nib.load(MAG_PATH).get_fdata()[..., 0, :] # remove extra dimension with size 1
    mask = nib.load(MASK_PATH).get_fdata()[..., 0] # remove extra dimension with size 1

    with open(FNAME_JSON) as json_file:
        json_data = json.load(json_file)

    pmu = PmuResp(PMU_PATH)
    pmu.read_resp
    acq_timestamps = get_acquisition_times(field_map_nib, json_data)

    if FIELD_MAP_PATH == FIELD_MAP_PATH_1:
        field_map_data = field_map_data[:, :, 4:24]
        mag = mag[:, :, 4:24]
        acq_timestamps = acq_timestamps[4:24]

    acq_pressures = pmu.interp_resp_trace(acq_timestamps)
    acq_timestamps = (acq_timestamps - acq_timestamps[0]) / 1000
    acq_pressures = acq_pressures - np.mean(acq_pressures)

    # Sort field map data based on acq_timestamps
    idx = np.argsort(acq_pressures, axis=0)
    acq_timestamps = np.take_along_axis(acq_timestamps, idx, axis=0)
    acq_pressures = np.take_along_axis(acq_pressures, idx, axis=0)
    field_map_data = np.take_along_axis(field_map_data, idx[..., np.newaxis].T, axis=-1)
    return field_map_data, mag, json_data, mask, acq_timestamps, acq_pressures


def prep_fm(fm, mag, mask, threshold):
    # Mask fm based on magnitude threshold
    mag_masked = mag.copy()
    fm_masked = fm.copy()
    mag_mask = mag > threshold
    mag_masked[~mag_mask] = np.nan
    mag_masked[~(mask==1)] = np.nan
    fm_mask = np.isnan(mag_masked)
    fm_masked[fm_mask] = np.nan

    # Get crop idexes
    x_min = np.min(np.where(fm_mask[..., 0] == False)[0])
    x_max = np.max(np.where(fm_mask[..., 0] == False)[0])
    y_min = np.min(np.where(fm_mask[..., 0] == False)[1])
    y_max = np.max(np.where(fm_mask[..., 0] == False)[1])

    # Crop
    fm_masked = fm_masked[x_min:x_max, y_min:y_max, :]

    # Remove temporal mean
    fm_masked = fm_masked - np.nanmean(fm_masked, axis=-1, keepdims=True)

    return fm_masked


def first_degree_polynomial(xy, a, b, c):
    x, y = xy
    return a + b*x + c*y


def second_degree_polynomial(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y


def third_degree_polynomial(xy, a, b, c, d, e, f, g, h, i):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + i*x*y**2


def fit_surface(data, plot = False):
    # Select func
    func = second_degree_polynomial
    # Get x, y, z points
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z_points = data.flatten()
    x_points = x.flatten()[~np.isnan(z_points)]
    y_points = y.flatten()[~np.isnan(z_points)]
    z_points = z_points[~np.isnan(z_points)]

    # Fit curve
    popt, pcov = opt.curve_fit(func, (x_points, y_points), z_points)
    X = np.arange(0, data.shape[1], 1)
    Y = np.arange(0, data.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Z = func((X, Y), *popt)
    Z[np.isnan(data)] = np.nan

    # Plot in 3D
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.scatter(x_points, y_points, z_points, c='r', marker='o')
        plt.show()
    return Z


def fit_surface_RIDGE(data, plot = False):
    # Select func
    func = second_degree_polynomial
    # Get x, y, z points
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z_points = data.flatten()
    x_points = x.flatten()[~np.isnan(z_points)]
    y_points = y.flatten()[~np.isnan(z_points)]
    z_points = z_points[~np.isnan(z_points)]

    degree = 2
    poly = PolynomialFeatures(degree)
    X_transformed = poly.fit_transform(np.vstack((x_points, y_points)).T)

    alpha = 0.01 # Regularization strength (hyperparameter to be tuned)
    ridge_model = Ridge(alpha=alpha, fit_intercept=False)  # Choose fit_intercept based on your data
    ridge_model.fit(X_transformed, z_points)

    # Predict using the RIDGE model
    X_mesh, Y_mesh = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[0], 1))
    XY = np.vstack((X_mesh.flatten(), Y_mesh.flatten())).T
    XY_transformed = poly.fit_transform(XY)
    Z_ridge = ridge_model.predict(XY_transformed)
    Z_ridge = Z_ridge.reshape(X_mesh.shape)

    if plot:
        plot_surface(z_points, x_points, y_points, X_mesh, Y_mesh, Z_ridge)

    Z_ridge[np.isnan(data)] = np.nan
    return Z_ridge


def fit_surface_LASSO(data, plot = False):
    # Get x, y, z points
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z_points = data.flatten()
    x_points = x.flatten()[~np.isnan(z_points)]
    y_points = y.flatten()[~np.isnan(z_points)]
    z_points = z_points[~np.isnan(z_points)]

    degree = 2
    poly = PolynomialFeatures(degree)
    X_transformed = poly.fit_transform(np.vstack((x_points, y_points)).T)

    alpha = 0.1  # Regularization strength (hyperparameter to be tuned)
    lasso_model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)  # Choose fit_intercept based on your data
    lasso_model.fit(X_transformed, z_points)

    # Predict using the LASSO model
    X_mesh, Y_mesh = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[0], 1))
    XY = np.vstack((X_mesh.flatten(), Y_mesh.flatten())).T
    XY_transformed = poly.fit_transform(XY)
    Z_lasso = lasso_model.predict(XY_transformed)
    Z_lasso = Z_lasso.reshape(X_mesh.shape)
    Z_lasso[np.isnan(data)] = np.nan
    # Plot in 3D
    if plot:
        plot_surface(z_points, x_points, y_points, X_mesh, Y_mesh, Z_lasso)

    return Z_lasso


def plot_surface(z_points, x_points, y_points, X_mesh, Y_mesh, Z_lasso):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_lasso)
    ax.scatter(x_points, y_points, z_points, c='r', marker='o')
    # Set axis titles
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('B0')
    plt.show()

    plt.imshow(Z_lasso)
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()


def fit_realtime_surface(data, pressures, fit):
    # Create new data
    surfaces = np.zeros_like(data)

    # Fit surface for every timepoint
    for t in range(data.shape[-1]):
        if t == 12:
            surfaces[..., t] = fit(data[..., t], plot=True)
        else:
            surfaces[..., t] = fit(data[..., t])

    # Plot AP mean through z direction for every time point
    max = int(np.nanmax(pressures))
    min = int(np.nanmin(pressures))
    print(min, max)
    colors = plt.cm.viridis(np.linspace(0, 1, max - min + 1))
    for t in range(data.shape[-1]):
        c = int(pressures[t][0] - min)
        plt.plot(np.nanmean(surfaces[..., t], axis=0), color=colors[c], label='t = ' + str(t))

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min, vmax=max))
    sm.set_array([])  # fake up the array of the scalar mappable
    plt.colorbar(sm, label='Pressure')
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # im1 = ax1.imshow(data[..., 0])
    # im2 = ax2.imshow(surfaces[..., 0])

    # fig.colorbar(im1, ax=ax1)
    # fig.colorbar(im2, ax=ax2)
    # plt.tight_layout()
    # plt.show()


def main():
    field_map, mag, json_data, mask, acq_timestamps, acq_pressures = \
        load_data(FIELD_MAP_PATH_1, MAG_PATH_1, FNAME_JSON_1, MASK_PATH, PMU_PATH)

    fm_masked = prep_fm(field_map, mag, mask, 200)
    fit_realtime_surface(fm_masked, acq_pressures, fit_surface_LASSO)


if __name__ == '__main__':
    main()
