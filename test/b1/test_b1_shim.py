#!usr/bin/env python3
# -*- coding: utf-8

import pathlib
import pytest
import tempfile

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.b1.b1_shim import *
from shimmingtoolbox.b1.load_vop import *
from shimmingtoolbox.load_nifti import read_nii


logging.basicConfig(level=logging.INFO)

fname_b1 = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'TB1map_axial.nii.gz')
_, _, b1_maps = read_nii(fname_b1, auto_scale=True)
mask = b1_maps.sum(axis=-1) != 0
cp_weights = np.asarray([0.3536, -0.3527+0.0247j, 0.2748-0.2225j, -0.1926-0.2965j, -0.3535+0.0062j, 0.2931+0.1977j,
                         0.3381+0.1034j, -0.1494+0.3204j])

path_sar_file = os.path.join(__dir_testing__, 'b1_maps', 'vop', 'SarDataUser.mat')
vop = load_siemens_vop(path_sar_file)


def test_b1_shim(caplog):
    shim_weights = b1_shim(b1_maps)
    assert r"No Q matrix provided, performing unconstrained optimization." in caplog.text
    assert r"No mask provided, masking all zero-valued pixels." in caplog.text
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"

def test_b1_shim_algo_2(caplog):
    shim_weights = b1_shim(b1_maps, mask, algorithm=2, target=20)
    assert r"No Q matrix provided, performing unconstrained optimization." in caplog.text
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_algo_2_no_target():
    with pytest.raises(ValueError, match=r"Algorithm 2 requires a target B1 value in nT/V."):
        b1_shim(b1_maps, mask, algorithm=2)


def test_b1_shim_algo_3(caplog):
    shim_weights = b1_shim(b1_maps, mask, algorithm=3)
    assert r"No Q matrix provided, performing unconstrained optimization." in caplog.text
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_constrained():
    shim_weights = b1_shim(b1_maps, mask, q_matrix=vop)
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_constrained_factor_too_small():
    with pytest.raises(ValueError, match=r"The SAR factor must be equal to or greater than 1."):
        b1_shim(b1_maps, mask, q_matrix=vop, SED=0.9)


def test_b1_shim_wrong_ndim():
    with pytest.raises(ValueError, match=r"The provided B1 maps have an unexpected number of dimensions.\nExpected: 4\n"
                                         r"Actual: 3"):
        b1_shim(b1_maps[:, :, :, 0], mask)


def test_b1_shim_wrong_mask_shape():
    with pytest.raises(ValueError, match=r"Mask and maps dimensions not matching.\n"
                                         r"Maps dimensions: \(64, 44, 5\)\n"
                                         r"Mask dimensions: \(63, 44, 5\)"):
        b1_shim(b1_maps, mask[:-1, :, :])


def test_b1_shim_cp_mode():
    shim_weights = b1_shim(b1_maps, mask, cp_weights)
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_cp_mode_not_normalized(caplog):
    cp_weights_not_normalized = np.asarray([2*cp_weights[i] for i in range(len(cp_weights))])
    shim_weights = b1_shim(b1_maps, mask, cp_weights_not_normalized)
    assert r"Normalizing the CP mode weights." in caplog.text
    assert len(shim_weights) == b1_maps.shape[3], "The number of shim weights does not match the number of coils"


def test_b1_shim_cp_mode_wrong_length():
    with pytest.raises(ValueError, match=r"The number of CP weights does not match the number of channels.\n"
                                         r"Number of CP weights: 7\n"
                                         r"Number of channels: 8"):
        b1_shim(b1_maps, mask, cp_weights[:-1])


def test_b1_shim_output_figure(caplog):
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        b1_shim(b1_maps, mask, path_output=tmp)


def test_vector_to_complex():
    assert np.isclose(vector_to_complex(np.asarray([1, 1, 1, 0, np.pi/2, np.pi])), np.asarray([1,  1j, -1])).all(),\
        "The function vector_to_complex returns unexpected results"


def test_vector_to_complex_wrong_length():
    with pytest.raises(ValueError, match=r"The vector must have an even number of elements."):
        vector_to_complex(np.asarray([1, 1, 1, 0, np.pi/2])), np.asarray([1,  1j, -1])


def test_complex_to_vector():
    assert np.isclose(complex_to_vector(np.asarray([1,  1j, -1])), np.asarray([1, 1, 1, 0, np.pi/2, np.pi])).all(),\
        "The function complex_to_vector returns unexpected results"


def test_combine_maps():
    dummy_weights = np.asarray([1+8j, 3-5j, 8+1j, 4-4j, 5-6j, 2-9j, 3+2j, 4-7j])
    combined_map = combine_maps(b1_maps, dummy_weights)
    values = [combined_map[30, 30, 0], combined_map[20, 30, 2], combined_map[40, 15, 3], combined_map[30, 40, 4]]
    values_to_match = [329.7406895062313, 237.49011367537193, 239.36672012680663, 209.72138461426627]
    assert np.isclose(values, values_to_match).all()


def test_combine_maps_wrong_weights_number():
    dummy_weights = np.asarray([1+8j, 3-5j, 8+1j, 4-4j, 5-6j, 2-9j, 3+2j])
    with pytest.raises(ValueError, match=f"The number of shim weights does not match the number of channels.\n"
                                         f"Number of shim weights: 7\n"
                                         f"Number of channels: 8"):
        combine_maps(b1_maps, dummy_weights)


def test_calc_cp_no_size(caplog):
    calc_cp(b1_maps, voxel_position=(32, 22, 2))
    assert r"No voxel size provided for CP computation. Default size set to (5, 5, 1)." in caplog.text


def test_calc_cp_excessive_size():
    with pytest.raises(ValueError, match=r"The size of the voxel used for CP computation exceeds the size of the B1 "
                                         r"maps.\n"
                                         r"B1 maps size: \(64, 44, 5\)\n"
                                         r"Voxel size: \(70, 32, 8\)"):
        calc_cp(b1_maps, voxel_size=(70, 32, 8))


def test_calc_cp_no_position(caplog):
    calc_cp(b1_maps, voxel_size=(10, 10, 2))
    assert r"No voxel position provided for CP computation. Default set to the center of the B1 maps." in caplog.text


def test_calc_cp_small_b1(caplog):
    with pytest.raises(ValueError, match=r"Provided B1 maps are too small to compute CP phases.\n"
                                         r"Minimum size: \(5, 5, 1\)\n"
                                         r"Actual size: \(4, 44, 5\)"):
        calc_cp(b1_maps[:4, ...])


def test_calc_cp_out_of_bounds_position():
    with pytest.raises(ValueError, match=r"The position of the voxel used to compute the CP mode exceeds the B1 maps "
                                         r"bounds.\n"
                                         r"B1 maps size: \(64, 44, 5\)\n"
                                         r"Voxel position: \(70, 32, 3\)"):
        calc_cp(b1_maps, voxel_position=(70, 32, 3))


def test_calc_cp_out_out_of_bounds_voxel():
    with pytest.raises(ValueError, match=r"Voxel bounds exceed the B1 maps."):
        calc_cp(b1_maps, voxel_size=(20, 10, 2), voxel_position=(55, 32, 3))


def test_calc_approx_cp():
    approx_weights = calc_approx_cp(8)
    assert np.isclose(approx_weights,
                      np.asarray([3.53553391e-01+0.00000000e+00j,  2.50000000e-01-2.50000000e-01j,
                                 2.16489014e-17-3.53553391e-01j, -2.50000000e-01-2.50000000e-01j,
                                 -3.53553391e-01-4.32978028e-17j, -2.50000000e-01+2.50000000e-01j,
                                 -6.49467042e-17+3.53553391e-01j,  2.50000000e-01+2.50000000e-01j])).all()
