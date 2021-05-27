#!usr/bin/env python3
# coding: utf-8

import os
import nibabel as nib
import json
import numpy as np
import pathlib
import tempfile

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.masking.shapes import shapes
from shimmingtoolbox.pmu import PmuResp
from shimmingtoolbox.shim.realtime_shim import realtime_shim


class TestRealtimeShim(object):
    def setup(self):
        # Fieldmap
        fname_fieldmap = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                      'sub-example_fieldmap.nii.gz')
        nii_fieldmap = nib.load(fname_fieldmap)
        self.nii_fieldmap = nii_fieldmap

        # anat image
        fname_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat',
                                  'sub-example_unshimmed_e1.nii.gz')
        nii_anat = nib.load(fname_anat)
        self.nii_anat = nii_anat

        # Set up mask
        # static
        nx, ny, nz = nii_anat.shape
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=30, len_dim2=30, len_dim3=nz)

        nii_mask_static = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        self.nii_mask_static = nii_mask_static

        # Riro
        mask = shapes(nii_anat.get_fdata(), 'cube',
                      center_dim1=int(nx / 2),
                      center_dim2=int(ny / 2),
                      len_dim1=30, len_dim2=30, len_dim3=nz)

        nii_mask_riro = nib.Nifti1Image(mask.astype(int), nii_anat.affine)
        self.nii_mask_riro = nii_mask_riro

        # Pmu
        fname_resp = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'PMUresp_signal.resp')
        pmu = PmuResp(fname_resp)
        self.pmu = pmu

        # Path for json file
        fname_json = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                  'sub-example_magnitude1.json')
        with open(fname_json) as json_file:
            json_data = json.load(json_file)

        self.json = json_data

    def test_default(self):
        """Test realtime_shim default parameters"""
        static_xcorrection, static_ycorrection, static_zcorrection, riro_xcorrection, riro_ycorrection, riro_zcorrection, mean_p, pressure_rms = realtime_shim(self.nii_fieldmap,
                                                                                  self.nii_anat,
                                                                                  self.pmu,
                                                                                  self.json)
        assert np.isclose(static_correction[0], 0.1291926595061463,)
        assert np.isclose(riro_correction[0], -0.00802980555042238)
        assert np.isclose(mean_p, 1326.3179660207873)
        assert np.isclose(pressure_rms, 1493.9468284155396)

    def test_mask(self):
        """Test realtime_shim mask parameter"""
        static_xcorrection, static_ycorrection, static_zcorrection, riro_xcorrection, riro_ycorrection, riro_zcorrection, mean_p, pressure_rms = realtime_shim(self.nii_fieldmap,
                                                                                  self.nii_anat,
                                                                                  self.pmu,
                                                                                  self.json,
                                                                                  nii_mask_anat_static=
                                                                                  self.nii_mask_static,
                                                                                  nii_mask_anat_riro=
                                                                                  self.nii_mask_riro)
        assert np.isclose(static_zcorrection[0], 0.2766538103967352)
        assert np.isclose(riro_zcorrection[0], -0.051144561917725075)
        assert np.isclose(mean_p, 1326.318)
        assert np.isclose(pressure_rms, 1493.9468284155396)

    def test_output_figure(self):
        """Test realtime_shim output figures parameter"""
        with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
            _, _, _, _ = realtime_shim(self.nii_fieldmap, self.nii_anat, self.pmu, self.json,
                                        nii_mask_anat_static=self.nii_mask_static,
                                        nii_mask_anat_riro=self.nii_mask_riro,
                                        path_output=tmp)

            assert len(os.listdir(tmp)) != 0

    # Tests that should throw errors
    def test_wrong_dim_fieldmap(self):
        """Wrong number of fieldmap dimensions."""

        fieldmap = self.nii_fieldmap.get_fdata()
        nii_fieldmap_3d = nib.Nifti1Image(fieldmap[..., 0], self.nii_fieldmap.affine)

        # This should return an error
        try:
            realtime_shim(nii_fieldmap_3d, self.nii_anat, self.pmu, self.json)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nWrong number of dimensions for fieldmap but does not throw an error.")
        assert False

    def test_wrong_dim_anat(self):
        """Wrong number of anat dimensions."""

        anat = self.nii_anat.get_fdata()
        nii_anat_2d = nib.Nifti1Image(anat[..., 0], self.nii_fieldmap.affine)

        # This should return an error
        try:
            realtime_shim(self.nii_fieldmap, nii_anat_2d, self.pmu, self.json)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nWrong number of dimensions for anat but does not throw an error.")
        assert False

    def test_wrong_dim_mask_static(self):
        """Wrong number of static mask dimensions."""

        mask = self.nii_mask_static.get_fdata()
        nii_mask_2d = nib.Nifti1Image(mask[..., 0], self.nii_fieldmap.affine)

        # This should return an error
        try:
            realtime_shim(self.nii_fieldmap, self.nii_anat, self.pmu, self.json, nii_mask_anat_static=nii_mask_2d)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nWrong number of dimensions for static mask but does not throw an error.")
        assert False

    def test_wrong_dim_mask_riro(self):
        """Wrong number of riro mask dimensions."""

        mask = self.nii_mask_riro.get_fdata()
        nii_mask_2d = nib.Nifti1Image(mask[..., 0], self.nii_fieldmap.affine)

        # This should return an error
        try:
            realtime_shim(self.nii_fieldmap, self.nii_anat, self.pmu, self.json, nii_mask_anat_riro=nii_mask_2d)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nWrong number of dimensions for riro mask but does not throw an error.")
        assert False
