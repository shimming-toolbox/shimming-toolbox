# coding: utf-8

import pytest
import os
from pathlib import Path
import json
import numpy as np
from shimmingtoolbox.simulate import *
from phantominator import shepp_logan


class TestCore(object):
    def setup(self):
        self.testFileName = "test"
        self.testFileNameNii = "test.nii"
        self.testFileNameMat = "test.mat"

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
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    # --------------Shepp-Logan type tests instance test-------------- #
    def test_shepplogan_init_returns_expected_starting_volume(self):
        test_obj = NumericalModel(model="shepp-logan")

        expected_volume = shepp_logan(128)
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepplogan_dims_init_returns_expected_starting_volume(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        expected_volume = shepp_logan(dims)
        actual_volume = test_obj.starting_volume

        np.testing.assert_array_equal(actual_volume, expected_volume)

    def test_shepp_logan_defines_expected_t2star_values(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        assert np.all(
            test_obj.volume["T2_star"][abs(test_obj.starting_volume - 0.2) < 0.001]
            == test_obj.T2_star["WM"]
        )
        assert np.all(
            test_obj.volume["T2_star"][abs(test_obj.starting_volume - 0.3) < 0.001]
            == test_obj.T2_star["GM"]
        )
        assert np.all(
            test_obj.volume["T2_star"][abs(test_obj.starting_volume - 1) < 0.001]
            == test_obj.T2_star["CSF"]
        )
        assert np.all(
            test_obj.volume["T2_star"][
                np.logical_and(
                    (abs(test_obj.starting_volume) < 0.0001),
                    test_obj.starting_volume != 0,
                )
            ]
            == test_obj.T2_star["WM"] / 2
        )
        assert np.all(
            test_obj.volume["T2_star"][abs(test_obj.starting_volume - 0.1) < 0.001]
            == (test_obj.T2_star["GM"] + test_obj.T2_star["WM"]) / 2
        )
        assert np.all(
            test_obj.volume["T2_star"][abs(test_obj.starting_volume - 0.4) < 0.001]
            == test_obj.T2_star["GM"] * 1.5
        )

    def test_shepp_logan_defines_expected_magnitude_values(self):
        dims = 256
        test_obj = NumericalModel(model="shepp-logan", num_vox=dims)

        assert np.all(
            test_obj.volume["proton_density"][
                abs(test_obj.starting_volume - 0.2) < 0.001
            ]
            == test_obj.proton_density["WM"]
        )
        assert np.all(
            test_obj.volume["proton_density"][
                abs(test_obj.starting_volume - 0.3) < 0.001
            ]
            == test_obj.proton_density["GM"]
        )
        assert np.all(
            test_obj.volume["proton_density"][abs(test_obj.starting_volume - 1) < 0.001]
            == test_obj.proton_density["CSF"]
        )
        assert np.all(
            test_obj.volume["proton_density"][
                np.logical_and(
                    (abs(test_obj.starting_volume) < 0.0001),
                    test_obj.starting_volume != 0,
                )
            ]
            == test_obj.proton_density["WM"] / 2
        )
        assert np.all(
            test_obj.volume["proton_density"][
                abs(test_obj.starting_volume - 0.1) < 0.001
            ]
            == (test_obj.proton_density["GM"] + test_obj.proton_density["WM"]) / 2
        )
        assert np.all(
            test_obj.volume["proton_density"][
                abs(test_obj.starting_volume - 0.4) < 0.001
            ]
            == test_obj.proton_density["GM"] * 1.5
        )

    # --------------generate_deltaB0 method tests-------------- #
    def test_generate_deltaB0_linear_floor_value(self):

        test_obj = NumericalModel(model="shepp-logan")

        m = 0
        b = 2
        test_obj.generate_deltaB0("linear", [m, b])

        deltaB0_map = test_obj.deltaB0

        assert np.allclose(
            np.mean(deltaB0_map[:]), b / (test_obj.gamma / (2 * np.pi)), rtol=10 ** -6
        )

    def test_generate_deltaB0_linear_slope_value(self):
        test_obj = NumericalModel(model="shepp-logan")

        m = 1
        b = 0
        test_obj.generate_deltaB0("linear", [m, b])

        deltaB0_map = test_obj.deltaB0

        dims = deltaB0_map.shape
        [X, Y] = np.meshgrid(
            np.linspace(-dims[0], dims[0], dims[0]),
            np.linspace(-dims[1], dims[1], dims[1]),
        )

        assert np.allclose(
            deltaB0_map[int(dims[0] / 2), int(dims[0] / 4)],
            m * X[int(dims[0] / 2), int(dims[0] / 4)] / (test_obj.gamma / (2 * np.pi)),
        )

    # --------------simulate_signal method tests-------------- #
    def test_simulate_signal_returns_expected_volume_size(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        expectedDims = (128, 128, 1, len(TE))
        actualDims = test_obj.measurement.shape

        assert actualDims == expectedDims

    def test_simulate_signal_get_returns_volume_of_expected_datatype(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        assert np.isreal(test_obj.get_magnitude().all())
        assert np.isreal(test_obj.get_phase().all())
        assert np.isreal(test_obj.get_real().all())
        assert np.isreal(test_obj.get_imaginary().all())

    def test_simulate_signal_SNR_results_in_noisy_backgroun(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]
        SNR = 50

        test_obj.simulate_measurement(FA, TE, SNR)

        magnitude_data = test_obj.get_magnitude()

        vecMagnitudeROI = magnitude_data[0:10, 0:10, 0, 0]
        vecMagnitudeROI = vecMagnitudeROI[:]

        assert np.std(vecMagnitudeROI) != 0

        phase_data = test_obj.get_phase()
        vecPhaseROI = phase_data[0:10, 0:10, 0, 0]
        vecPhaseROI = vecPhaseROI[:]

        assert np.std(vecPhaseROI) != 0

    def test_simulate_signal_dual_echo_calculates_expected_B0(self):
        test_obj = NumericalModel(model="shepp-logan")

        B0_hz = 13
        test_obj.generate_deltaB0("linear", [0.0, B0_hz])
        TR = 0.025
        TE = [0.004, 0.008]
        test_obj.simulate_measurement(TR, TE)
        phaseMeas = test_obj.get_phase()

        phaseTE1 = np.squeeze(phaseMeas[:, :, 0, 0])
        phaseTE2 = np.squeeze(phaseMeas[:, :, 0, 1])

        B0_meas = (phaseTE2[63, 63] - phaseTE1[63, 63]) / (TE[1] - TE[0])
        B0_meas_hz = B0_meas / (2 * np.pi)

        assert np.allclose(B0_hz, B0_meas_hz, rtol=10 ** -6)

    def test_simulate_signal_dual_echo_righthand_calculates_negative_B0(self):
        test_obj = NumericalModel(model="shepp-logan")
        test_obj.handedness = "right"

        B0_hz = 13
        test_obj.generate_deltaB0("linear", [0.0, B0_hz])
        TR = 0.025
        TE = [0.004, 0.008]
        test_obj.simulate_measurement(TR, TE)
        phaseMeas = test_obj.get_phase()

        phaseTE1 = np.squeeze(phaseMeas[:, :, 0, 0])
        phaseTE2 = np.squeeze(phaseMeas[:, :, 0, 1])

        B0_meas = (phaseTE2[63, 63] - phaseTE1[63, 63]) / (TE[1] - TE[0])
        B0_meas_hz = B0_meas / (2 * np.pi)

        assert np.allclose(B0_hz, -B0_meas_hz, rtol=10 ** -6)

    # --------------save method tests-------------- #

    def test_save_nii(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        test_obj.save("Magnitude", self.testFileName)
        # Default option for save is a NIfTI output.

        assert os.path.isfile(Path(self.testFileName + ".nii"))

        # Verify that JSON was written correctly
        assert os.path.isfile(Path(self.testFileName + ".json"))

        fname = Path(self.testFileName + ".json")

        with open(str(fname)) as f:
            data = json.load(f)

        np.testing.assert_equal(data["EchoTime"], TE)
        np.testing.assert_equal(data["FlipAngle"], FA)

        if os.path.isfile(Path(self.testFileName + ".nii")):
            os.remove(str(Path(self.testFileName + ".nii")))

        if os.path.isfile(Path(self.testFileName + ".json")):
            os.remove(str(Path(self.testFileName + ".json")))

    def test_save_nii_with_extension(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        test_obj.save("Magnitude", self.testFileNameNii)
        # Default option for save is a NIfTI output.

        assert os.path.isfile(Path(self.testFileNameNii))

        # Verify that JSON was written correctly
        assert os.path.isfile(Path(self.testFileName + ".json"))

        fname = Path(self.testFileName + ".json")

        with open(str(fname)) as f:
            data = json.load(f)

        np.testing.assert_equal(data["EchoTime"], TE)
        np.testing.assert_equal(data["FlipAngle"], FA)

        if os.path.isfile(Path(self.testFileNameNii)):
            os.remove(str(Path(self.testFileNameNii)))

        if os.path.isfile(Path(self.testFileName + ".json")):
            os.remove(str(Path(self.testFileName + ".json")))

    def test_save_mat(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        test_obj.save("Magnitude", self.testFileName, "mat")

        assert os.path.isfile(Path(self.testFileName + ".mat"))

        # Verify that JSON was written correctly
        assert os.path.isfile(Path(self.testFileName + ".json"))

        fname = Path(self.testFileName + ".json")

        with open(str(fname)) as f:
            data = json.load(f)

        np.testing.assert_equal(data["EchoTime"], TE)
        np.testing.assert_equal(data["FlipAngle"], FA)

        if os.path.isfile(Path(self.testFileName + ".mat")):
            os.remove(str(Path(self.testFileName + ".mat")))

        if os.path.isfile(Path(self.testFileName + ".json")):
            os.remove(str(Path(self.testFileName + ".json")))

    def test_save_mat_with_extension(self):
        test_obj = NumericalModel(model="shepp-logan")

        FA = 15
        TE = [0.003, 0.015]

        test_obj.simulate_measurement(FA, TE)

        test_obj.save("Magnitude", self.testFileNameMat, "mat")

        assert os.path.isfile(Path(self.testFileNameMat))

        # Verify that JSON was written correctly
        assert os.path.isfile(Path(self.testFileName + ".json"))

        fname = Path(self.testFileName + ".json")

        with open(str(fname)) as f:
            data = json.load(f)

        np.testing.assert_equal(data["EchoTime"], TE)
        np.testing.assert_equal(data["FlipAngle"], FA)

        if os.path.isfile(Path(self.testFileNameMat)):
            os.remove(str(Path(self.testFileNameMat)))

        if os.path.isfile(Path(self.testFileName + ".json")):
            os.remove(str(Path(self.testFileName + ".json")))

    # --------------generate_signal method tests-------------- #

    def test_generate_signal_case_1(self):
        protonDensity = 80

        T2star = 100
        FA = 90
        TE = 0
        deltaB0 = 0
        gamma = 42.58 * 10 ** 6
        handedness = "left"

        actual_signal = NumericalModel.generate_signal(
            protonDensity, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = protonDensity

        assert actual_signal == expected_signal

    def test_generate_signal_case_2(self):

        FA = 0

        protonDensity = 80
        T2star = 100
        TE = 0
        deltaB0 = 0
        gamma = 42.58 * 10 ** 6
        handedness = "left"

        actual_signal = NumericalModel.generate_signal(
            protonDensity, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = 0

        assert actual_signal == expected_signal

    def test_generate_signal_case_3(self):

        FA = 20
        protonDensity = 80
        T2star = 100
        TE = 0.010
        deltaB0 = 2
        gamma = 42.58 * 10 ** 6
        handedness = "left"

        if handedness == "left":
            sign = -1
        elif handedness == "right":
            sign = 1

        actual_signal = NumericalModel.generate_signal(
            protonDensity, T2star, FA, TE, deltaB0, gamma, handedness
        )
        expected_signal = (
            protonDensity
            * np.sin(np.deg2rad(FA))
            * np.exp(-TE / T2star - sign * 1j * gamma * deltaB0 * TE)
        )

        assert ~np.isreal(expected_signal)
        assert actual_signal == expected_signal
