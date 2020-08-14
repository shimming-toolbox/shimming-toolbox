# coding: utf-8
"""Create numerical model data for multi-echo B0 field mapping data

This module is for numerically simulating multi-echo B0 field mapping data. It
considers features like: background B0 field, flip angle, echo time, and noise.

Typical usage example:

::

    from shimmingtoolbox.simulate import *

    b0_sim = NumericalModel(model="shepp-logan")

    # Generate a background B0
    b0_field = 13 # (Hz)
    b0_sim.generate_deltaB0("linear", [0.0, b0_field])

    # Simulate the signal data
    FA = 15 # (degrees)
    TE = [0.003, 0.015] # (seconds)
    SNR = 50
    b0_sim.simulate_measurement(FA, TE, SNR)

    # Save simulation as NIfTI file (JSON sidecar also exported with parameters)
    b0_sim.save('Phase', 'b0_mapping_data.nii', format='nifti')
"""

import os, json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import savemat

from phantominator import shepp_logan

np.seterr(divide="ignore", invalid="ignore")


class NumericalModel:
    """Multi-echo B0 field mapping data numerical simulator.

    Simulate multi-echo B0 field mapping data in the presence of a B0 field. 
    Can simulate data under ideal conditions or with noise. Export simulations 
    in a NIfTI or ``.mat`` file formats.

    Attributes:
        gamma (float): Gyromagnetic ratio in rad * Hz / Tesla.
        field_strength (float): Static field strength in Tesla.
        handedness: Orientation of the cross-product for the Larmor equation. The value of this attribute is MRI vendor-dependent.
        measurement: Simulated measurement data array.
        proton_density: Default assumed brain proton density in %.
        T2_star: Default assumed brain T2* values in seconds at 3T.
    """

    # Gyromagnetic ratio in rad*Hz/Tesla
    gamma = 267.52218744 * 10 ** 6

    # Static field strength in Tesla
    field_strength = 3.0

    # Siemens & Canon = 'left', GE & Philips = 'right'
    handedness = "left"

    # Simulated measurement data. Dimensions are [x, y, z, TE].
    measurement = None

    # Default brain proton density in %
    proton_density = {"WM": 70, "GM": 82, "CSF": 100}

    # Default brain T2* values in seconds at 3T
    T2_star = {"WM": 0.053, "GM": 0.066, "CSF": 0.10}

    _num_vox = None
    _starting_volume = None
    _volume = {"T2_star": None, "proton_density": None}

    def __init__(self, model=None, num_vox=128):
        """Initializes a NumericalModel object.

        Defines the starting volume. Sets the background B0 field to zeros.

        Args:
            model: Volume model used for the measurement simulation. Default is no
                object (zeros). Implemented models are: 'shepp-logan'.
            num_vox: In-plane dimensions of the simulated - square.

        Returns:
            NumericalModel class object with no background B0 field or simulated
            measurements.
        """

        self._num_vox = num_vox

        if model is None:
            self._starting_volume = np.zeros((num_vox, num_vox))
        elif model == "shepp-logan":
            self._shepp_logan_brain(num_vox)

        self.deltaB0 = self._starting_volume * 0

    def generate_deltaB0(self, field_type, params):
        """Generates a background B0 field.

        Defines the starting volume. Sets the background B0 field to zeros.

        Args:
            field_type: Type of field to be generated. Available implementations 
                are: 'linear'.
            params: Parameters defining the field for the selected field type.
                field_type = 'linear': [m b] where m (Hz/pixel) is the slope and b
                    is the floor field (Hz).
        """

        if field_type == "linear":
            m = params[0]
            b = params[1]

            dims = self._starting_volume.shape

            [X, Y] = np.meshgrid(
                np.linspace(-dims[0], dims[0], dims[0]),
                np.linspace(-dims[1], dims[1], dims[1]),
            )

            self.deltaB0 = m * X + b

            self.deltaB0 = self.deltaB0 / (self.gamma / (2 * np.pi))
        else:
            Exception("Undefined deltaB0 field type")

    def simulate_measurement(self, FA, TE, SNR=None):
        """Simulates a multi-echo measurement for field mapping

        Resets the measurement class attribute to zero before simulating. Simulates
        the signal for each echo-time provided. If defined, adds noise to the 
        complex simulated signal measurements using an SNR value.

        Args:
            FA: Flip angle in degrees.
            TE: Echo-times in seconds. Can be either a single value, list, or 
                array.
            SNR: Signal-to-noise ratio used to define noise. If not set, no noise
                is added to the measurements.
        """

        self.FA = FA
        self.TE = TE

        numTE = len(TE)
        vol_dims = self._starting_volume.shape

        if len(vol_dims) == 2:
            self.measurement = np.zeros(
                (vol_dims[0], vol_dims[1], 1, numTE), dtype="complex_"
            )
        elif len(vol_dims) == 3:
            self.measurement = np.zeros(
                (vol_dims[0], vol_dims[1], vol_dims[2], numTE), dtype="complex_"
            )

        for ii in range(0, numTE):
            if len(vol_dims) == 2:
                self.measurement[:, :, 0, ii] = self.generate_signal(
                    np.squeeze(self._volume["proton_density"]),
                    np.squeeze(self._volume["T2_star"]),
                    FA,
                    TE[ii],
                    np.squeeze(self.deltaB0),
                    self.gamma,
                    self.handedness,
                )
            else:
                self.measurement[:, :, :, ii] = self.generate_signal(
                    np.squeeze(self._volume["proton_density"]),
                    np.squeeze(self._volume["T2_star"]),
                    FA,
                    TE[ii],
                    np.squeeze(self.deltaB0),
                    self.gamma,
                    self.handedness,
                )

        if SNR is not None:
            self.measurement = NumericalModel.add_noise(self.measurement, SNR)

    @staticmethod
    def generate_signal(proton_density, T2_star, FA, TE, deltaB0, gamma, handedness):

        if handedness == "left":
            sign = -1
        elif "right":
            sign = 1

        deltaB0 = np.array(deltaB0)
        vol_dims = deltaB0.shape

        if len(vol_dims) == 2:
            signal = np.zeros((vol_dims[0], vol_dims[1], 1), dtype="complex_")
        elif len(vol_dims) == 3:
            signal = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2]), dtype="complex_")

        signal = (
            proton_density
            * np.sin(np.deg2rad(FA))
            * np.exp(-TE / T2_star - sign * 1j * gamma * deltaB0 * TE)
        )

        return signal

    @staticmethod
    def add_noise(volume, SNR):
        noiseSTD = np.max(volume[:]) / SNR

        vol_dims = volume.shape
        noisy_real = (
            np.real(volume)
            + np.random.randn(vol_dims[0], vol_dims[1], vol_dims[2], vol_dims[3])
            * noiseSTD
        )
        noisy_imaginary = (
            np.imag(volume)
            + np.random.randn(vol_dims[0], vol_dims[1], vol_dims[2], vol_dims[3])
            * noiseSTD
        )

        noisy_volume = noisy_real + 1j * noisy_imaginary
        return noisy_volume

    def get_magnitude(self):
        return np.abs(self.measurement)

    def get_phase(self):
        return np.angle(self.measurement)

    def get_real(self):
        return np.real(self.measurement)

    def get_imaginary(self):
        return np.imag(self.measurement)

    def save(self, data_type, file_name, format=None):
        """Exports simulated data to a file with a JSON sidecar.

        Resets the measurement class attribute to zero before simulating. Simulates
        the signal for each echo-time provided. If defined, adds noise to the 
        complex simulated signal measurements using an SNR value.

        Args:
            data_type: Export data type. "Magnitude", "Phase", "Real", or
                "Imaginary".
            file_name: Filename of exported file, with or without file extension.
            format: File format for exported data. If no value given, will attempt
                to extract format from filename file extension, otherwise default
                to NIfTI.
        """
        if format is None:
            format = "nifti"

        if file_name[-4:] == ".nii":
            if format != "nifti":
                print("File extension and format do not match - saving to NIfTI format")
                format = "nifti"
            file_name = file_name[0:-4]
        elif file_name[-4:] == ".mat":
            if format != "mat":
                print("File extension and format do not match - saving to MAT format")
                format = "mat"
            file_name = file_name[0:-4]

        if data_type == "Magnitude":
            vol = self.get_magnitude()
        elif data_type == "Phase":
            vol = self.get_phase()
        elif data_type == "Real":
            vol = self.get_real()
        elif data_type == "Imaginary":
            vol = self.get_imaginary()
        else:
            Exception("Unknown datatype")

        if format == "nifti":
            empty_header = nib.Nifti1Header()

            img = nib.Nifti1Image(np.rot90(vol), affine=np.eye(4), header=empty_header)
            nib.save(img, Path(file_name + ".nii"))

            self._write_json(file_name)
        elif format == "mat":
            savemat(Path(file_name + ".mat"), {"vol": vol})
            self._write_json(file_name)

    def _customize_shepp_logan(self, volume, class1, class2, class3):

        custom_volume = deepcopy(volume)

        custom_volume[abs(volume - 0.2) < 0.001] = class1
        custom_volume[abs(volume - 0.3) < 0.001] = class2
        custom_volume[abs(volume - 1) < 0.001] = class3

        custom_volume[np.logical_and((abs(volume) < 0.0001), volume != 0)] = class1 / 2
        custom_volume[abs(volume - 0.1) < 0.001] = (class2 + class1) / 2
        custom_volume[abs(volume - 0.4) < 0.001] = class2 * 1.5

        return custom_volume

    def _shepp_logan_brain(self, numVox):
        self._starting_volume = shepp_logan(numVox)

        self._volume["proton_density"] = self._customize_shepp_logan(
            self._starting_volume,
            self.proton_density["WM"],
            self.proton_density["GM"],
            self.proton_density["CSF"],
        )
        self._volume["T2_star"] = self._customize_shepp_logan(
            self._starting_volume,
            self.T2_star["WM"],
            self.T2_star["GM"],
            self.T2_star["CSF"],
        )

    def _write_json(self, file_name):
        pulse_seq_properties = {"EchoTime": self.TE, "FlipAngle": self.FA}

        with open(Path(file_name + ".json"), "w") as outfile:
            json.dump(pulse_seq_properties, outfile)
