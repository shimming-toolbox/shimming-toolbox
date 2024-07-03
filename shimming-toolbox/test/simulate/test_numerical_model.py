# coding: utf-8

import json
import os
import numpy as np
import pathlib
from phantominator import shepp_logan
import tempfile

from shimmingtoolbox.simulate import NumericalModel


class TestNumericalModel(object):
    def setup_method(self):
        self.test_filename = "test"
        self.test_filename_nii = "test.nii"
        self.test_filename_mat = "test.mat"

    @classmethod
    def teardown_class(cls):
        pass

    # --------------class instance tests-------------- #
    def test_initiate_object_is_expected_class(self):

        test_obj = NumericalModel()

        assert isinstance(test_obj, NumericalModel)

    def test_empty_initialization_returns_expected_starting_volume(self):

        test_obj = NumericalModel()

        expected_volume = np.zeros((128, 128))
        actual_volume = test_obj._starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    # --------------Shepp-Logan type tests instance test-------------- #
    def test_shepplogan_init_returns_expected_starting_volume(self):
        test_obj = NumericalModel(model="shepp-logan")

        expected_volume = shepp_logan(128)
        actual_volume = test_obj._starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepplogan_dims_init_returns_expected_starting_volume(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        expected_volume = shepp_logan(dims)
        actual_volume = test_obj._starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepp_logan_defines_expected_t2star_values(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        assert np.all(
            test_obj._volume["T2_star"][abs(test_obj._starting_volume - 0.2) < 0.001]
            == test_obj.T2_star["WM"]
        )
        assert np.all(
            test_obj._volume["T2_star"][abs(test_obj._starting_volume - 0.3) < 0.001]
            == test_obj.T2_star["GM"]
        )
        assert np.all(
            test_obj._volume["T2_star"][abs(test_obj._starting_volume - 1) < 0.001]
            == test_obj.T2_star["CSF"]
        )
        assert np.all(
            test_obj._volume["T2_star"][
                np.logical_and(
                    (abs(test_obj._starting_volume) < 0.0001),
                    test_obj._starting_volume != 0,
                )
            ]
            == test_obj.T2_star["WM"] / 2
        )
        assert np.all(
            test_obj._volume["T2_star"][abs(test_obj._starting_volume - 0.1) < 0.001]
            == (test_obj.T2_star["GM"] + test_obj.T2_star["WM"]) / 2
        )
        assert np.all(
            test_obj._volume["T2_star"][abs(test_obj._starting_volume - 0.4) < 0.001]
            == test_obj.T2_star["GM"] * 1.5
        )

    def test_shepp_logan_defines_expected_magnitude_values(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        assert np.all(
            test_obj._volume["proton_density"][
                abs(test_obj._starting_volume - 0.2) < 0.001
            ]
            == test_obj.proton_density["WM"]
        )
        assert np.all(
            test_obj._volume["proton_density"][
                abs(test_obj._starting_volume - 0.3) < 0.001
            ]
            == test_obj.proton_density["GM"]
        )
        assert np.all(
            test_obj._volume["proton_density"][
                abs(test_obj._starting_volume - 1) < 0.001
            ]
            == test_obj.proton_density["CSF"]
        )
        assert np.all(
            test_obj._volume["proton_density"][
                np.logical_and(
                    (abs(test_obj._starting_volume) < 0.0001),
                    test_obj._starting_volume != 0,
                )
            ]
            == test_obj.proton_density["WM"] / 2
        )
        assert np.all(
            test_obj._volume["proton_density"][
                abs(test_obj._starting_volume - 0.1) < 0.001
            ]
            == (test_obj.proton_density["GM"] + test_obj.proton_density["WM"]) / 2
        )
        assert np.all(
            test_obj._volume["proton_density"][
                abs(test_obj._starting_volume - 0.4) < 0.001
            ]
            == test_obj.proton_density["GM"] * 1.5
        )

    # --------------generate_deltaB0 method tests-------------- #
    def test_generate_deltaB0_x_floor_value(self):

        test_obj = NumericalModel(model="shepp-logan")

        m = 0
        b = 2
        test_obj.generate_deltaB0("x", [m, b])

        deltaB0_map = test_obj.deltaB0

        assert np.allclose(
            np.mean(deltaB0_map), b / (test_obj.gamma / (2 * np.pi)), rtol=10 ** -6
        )

    def test_generate_deltaB0_y_floor_value(self):

        test_obj = NumericalModel(model="shepp-logan")

        m = 0
        b = 2
        test_obj.generate_deltaB0("y", [m, b])

        deltaB0_map = test_obj.deltaB0

        assert np.allclose(
            np.mean(deltaB0_map), b / (test_obj.gamma / (2 * np.pi)), rtol=10 ** -6
        )

    def test_generate_deltaB0_z_floor_value(self):

        test_obj = NumericalModel(model="shepp-logan")

        m = 0
        b = 2
        test_obj.generate_deltaB0("z", [m, b])

        deltaB0_map = test_obj.deltaB0

        assert np.allclose(
            np.mean(deltaB0_map), b / (test_obj.gamma / (2 * np.pi)), rtol=10 ** -6
        )

    def test_generate_deltaB0_x_slope_value(self):
        test_obj = NumericalModel(model="shepp-logan")

        m = 1
        b = 0
        test_obj.generate_deltaB0("x", [m, b])

        deltaB0_map = test_obj.deltaB0

        dims = deltaB0_map.shape
        [X, _] = np.meshgrid(
            np.linspace(-dims[0], dims[0], dims[0]),
            np.linspace(-dims[1], dims[1], dims[1]),
        )

        assert np.allclose(
            deltaB0_map[int(dims[0] / 2), int(dims[1] / 4)],
            m * X[int(dims[0] / 2), int(dims[1] / 4)] / (test_obj.gamma / (2 * np.pi)),
        )

    def test_generate_deltaB0_y_slope_value(self):
        test_obj = NumericalModel(model="shepp-logan")

        m = 1
        b = 0
        test_obj.generate_deltaB0("y", [m, b])

        deltaB0_map = test_obj.deltaB0

        dims = deltaB0_map.shape
        [_, Y] = np.meshgrid(
            np.linspace(-dims[0], dims[0], dims[0]),
            np.linspace(-dims[1], dims[1], dims[1]),
        )

        assert np.allclose(
            deltaB0_map[int(dims[0] / 2), int(dims[1] / 4)],
            m * Y[int(dims[0] / 2), int(dims[1] / 4)] / (test_obj.gamma / (2 * np.pi)),
        )

    def test_generate_deltaB0_z_slope_value(self):
        test_obj = NumericalModel(model="shepp-logan", n_slices=4)

        m = 1
        b = 0
        test_obj.generate_deltaB0("z", [m, b])

        deltaB0_map = test_obj.deltaB0

        dims = deltaB0_map.shape
        [_, _, Z] = np.meshgrid(
            np.linspace(-dims[0], dims[0], dims[0]),
            np.linspace(-dims[1], dims[1], dims[1]),
            np.linspace(-dims[2], dims[2], dims[2])
        )

        assert np.allclose(
            deltaB0_map[int(dims[0] / 2), int(dims[1] / 4)],
            m * Z[int(dims[0] / 2), int(dims[1] / 4)] / (test_obj.gamma / (2 * np.pi)),
        )

    # --------------simulate_signal method tests-------------- #
    def test_simulate_signal_returns_expected_volume_size(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        expected_dims = (128, 128, 1, len(TE))
        actual_dims = test_obj.measurement.shape

        assert actual_dims == expected_dims

    def test_simulate_signal_get_returns_volume_of_expected_datatype(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        assert np.isreal(test_obj.get_magnitude().all())
        assert np.isreal(test_obj.get_phase().all())
        assert np.isreal(test_obj.get_real().all())
        assert np.isreal(test_obj.get_imaginary().all())

    def test_simulate_signal_SNR_results_in_noisy_background(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]
        SNR = 50

        test_obj.simulate_measurement(FA, TE, SNR)

        magnitude_data = test_obj.get_magnitude()

        vec_magnitude_roi = magnitude_data[0:10, 0:10, 0, 0]
        vec_magnitude_roi = vec_magnitude_roi[:]

        assert np.std(vec_magnitude_roi) != 0

        phase_data = test_obj.get_phase()
        vec_phase_roi = phase_data[0:10, 0:10, 0, 0]
        vec_phase_roi = vec_phase_roi[:]

        assert np.std(vec_phase_roi) != 0

    def test_simulate_signal_dual_echo_calculates_expected_B0(self):
        test_obj = NumericalModel(model="shepp-logan")

        B0_hz = 13
        test_obj.generate_deltaB0("x", [0.0, B0_hz])
        TR = 0.025
        TE = [0.004, 0.008]
        test_obj.simulate_measurement(TR, TE)
        phase_meas = test_obj.get_phase()

        phase_TE1 = np.squeeze(phase_meas[:, :, 0, 0])
        phase_TE2 = np.squeeze(phase_meas[:, :, 0, 1])

        B0_meas = (phase_TE2[63, 63] - phase_TE1[63, 63]) / (TE[1] - TE[0])
        B0_meas_hz = B0_meas / (2 * np.pi)

        assert np.allclose(B0_hz, B0_meas_hz, rtol=10 ** -6)

    def test_simulate_signal_dual_echo_righthand_calculates_negative_B0(self):
        test_obj = NumericalModel(model="shepp-logan")
        test_obj.handedness = "right"

        B0_hz = 13
        test_obj.generate_deltaB0("x", [0.0, B0_hz])
        TR = 0.025
        TE = [0.004, 0.008]
        test_obj.simulate_measurement(TR, TE)
        phase_meas = test_obj.get_phase()

        phase_TE1 = np.squeeze(phase_meas[:, :, 0, 0])
        phase_TE2 = np.squeeze(phase_meas[:, :, 0, 1])

        B0_meas = (phase_TE2[63, 63] - phase_TE1[63, 63]) / (TE[1] - TE[0])
        B0_meas_hz = B0_meas / (2 * np.pi)

        assert np.allclose(B0_hz, -B0_meas_hz, rtol=10 ** -6)

    # --------------save method tests-------------- #

    def test_save_nii(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        # create temporary directory
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            test_obj.save("Magnitude", os.path.join(tmp, self.test_filename), manufacturer="Siemens")
            # Default option for save is a NIfTI output.

            assert os.path.isfile(os.path.join(tmp, self.test_filename + "_TE0" + ".nii"))
            assert os.path.isfile(os.path.join(tmp, self.test_filename + "_TE1" + ".nii"))

            fname0 = os.path.join(tmp, self.test_filename + "_TE0" + ".json")
            fname1 = os.path.join(tmp, self.test_filename + "_TE1" + ".json")
            # Verify that JSON was written correctly
            assert os.path.isfile(fname0)
            assert os.path.isfile(fname1)

            with open(str(fname0)) as f:
                data0 = json.load(f)
            with open(str(fname1)) as f:
                data1 = json.load(f)

            np.testing.assert_equal(data0["EchoTime"], TE[0])
            np.testing.assert_equal(data0["FlipAngle"], FA)
            np.testing.assert_equal(data0["Manufacturer"], 'Siemens')
            np.testing.assert_equal(data1["EchoTime"], TE[1])
            np.testing.assert_equal(data1["FlipAngle"], FA)
            np.testing.assert_equal(data1["Manufacturer"], 'Siemens')

    def test_save_nii_with_extension(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            # Default option for save is a NIfTI output.
            test_obj.save("Magnitude", os.path.join(tmp, self.test_filename_nii))

            assert os.path.isfile(os.path.join(tmp, self.test_filename + "_TE0" + ".nii"))

            fname = os.path.join(tmp, self.test_filename + "_TE0" + ".json")

            # Verify that JSON was written correctly
            assert os.path.isfile(fname)

            with open(str(fname)) as f:
                data = json.load(f)

            np.testing.assert_equal(data["EchoTime"], TE[0])
            np.testing.assert_equal(data["FlipAngle"], FA)
            np.testing.assert_equal(data["Manufacturer"], 'Simulated')

    def test_save_mat(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            test_obj.save("Magnitude", os.path.join(tmp, self.test_filename), "mat")

            assert os.path.isfile(os.path.join(tmp, self.test_filename + ".mat"))

            fname = os.path.join(tmp, self.test_filename + ".json")

            # Verify that JSON was written correctly
            assert os.path.isfile(fname)

            with open(str(fname)) as f:
                data = json.load(f)

            np.testing.assert_equal(data["EchoTime"], TE)
            np.testing.assert_equal(data["FlipAngle"], FA)

    def test_save_mat_with_extension(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            test_obj.save("Magnitude", os.path.join(tmp, self.test_filename_mat), "mat")

            assert os.path.isfile( os.path.join(tmp, self.test_filename_mat))

            fname = os.path.join(tmp, self.test_filename + ".json")

            # Verify that JSON was written correctly
            assert os.path.isfile(fname)

            with open(str(fname)) as f:
                data = json.load(f)

            np.testing.assert_equal(data["EchoTime"], TE)
            np.testing.assert_equal(data["FlipAngle"], FA)

    # --------------generate_signal method tests-------------- #

    def test_generate_signal_case_1(self):
        proton_density = 80

        T2star = 100
        FA = 90
        TE = 0
        deltaB0 = 0
        gamma = 42.58 * 10 ** 6
        handedness = "left"

        actual_signal = NumericalModel.generate_signal(
            proton_density, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = proton_density

        assert actual_signal == expected_signal

    def test_generate_signal_case_2(self):

        FA = 0

        proton_density = 80
        T2star = 100
        TE = 0
        deltaB0 = 0
        gamma = 42.58 * 10 ** 6
        handedness = "left"

        actual_signal = NumericalModel.generate_signal(
            proton_density, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = 0

        assert actual_signal == expected_signal

    def test_generate_signal_case_3(self):

        FA = 20
        proton_density = 80
        T2star = 100
        TE = 0.010
        deltaB0 = 2
        gamma = 42.58 * 10 ** 6
        handedness = "left"
        sign = -1

        actual_signal = NumericalModel.generate_signal(
            proton_density, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = (
            proton_density
            * np.sin(np.deg2rad(FA))
            * np.exp(-TE / T2star - sign * 1j * gamma * deltaB0 * TE)
        )

        assert ~np.isreal(expected_signal)
        assert actual_signal == expected_signal
