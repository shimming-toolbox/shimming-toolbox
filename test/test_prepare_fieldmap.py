#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import pytest
import math
import nibabel as nib
import numpy as np

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.prepare_fieldmap import prepare_fieldmap


@pytest.mark.prelude
class TestPrepareFieldmap(object):
    def setup(self):
        fname_phase = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                   'sub-example_phasediff.nii.gz')
        nii_phase = nib.load(fname_phase)
        self.phase = (nii_phase.get_fdata() * 2 * math.pi / 4095) - math.pi  # [-pi, pi]
        self.affine = nii_phase.affine
        fname_mag = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                 'sub-example_magnitude1.nii.gz')
        nii_mag = nib.load(fname_mag)
        self.mag = nii_mag.get_fdata()
        self.echo_times = [0.00246, 0.00492]

    def test_prepare_fieldmap_1_echo(self):
        """Test default works."""
        fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine)

        assert fieldmap.shape == self.phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        assert np.all(np.isclose(fieldmap[30:35, 40, 0, 0],
                                 np.array([18.51355514, 13.84794053,  9.48013154,  5.11232207,  0.64524454])))

    def test_prepare_fieldmap_with_mag(self):
        """Test mag works."""
        fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine, mag=self.mag)

        assert fieldmap.shape == self.phase.shape

    def test_prepare_fieldmap_2_echoes(self):
        """Test 2 echoes works."""

        # Import 2 echoes and rescale
        fname_phase1 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        nii_phase1 = nib.load(fname_phase1)
        phase1 = (nii_phase1.get_fdata() * 2 * math.pi / 4095) - math.pi
        fname_phase2 = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        nii_phase2 = nib.load(fname_phase2)
        phase2 = (nii_phase2.get_fdata() * 2 * math.pi / 4095) - math.pi

        # Load mag data to speed it prelude
        fname_mag = os.path.join(__dir_testing__, 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        mag = nib.load(fname_mag).get_fdata()

        echo_times = [0.0025, 0.0055]

        fieldmap = prepare_fieldmap([phase1, phase2], echo_times, nii_phase1.affine, mag=mag)

        assert fieldmap.shape == phase1.shape

    # Tests that should throw errors
    def test_prepare_fieldmap_wrong_range(self):
        """Test error when range is not between -pi and pi."""

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase - math.pi], self.echo_times, self.affine)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nRange is not between -pi and pi but does not throw an error.")
        assert False

    def test_prepare_fieldmap_wrong_echo_times(self):
        """Wrong number of echo times."""

        echo_times = [0.001, 0.002, 0.003]

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase], echo_times, self.affine)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nEcho_times has too many elements but does not throw an error.")
        assert False

    def test_prepare_fieldmap_mag_wrong_shape(self):
        """Mag has the wrong shape."""

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine, mag=np.zeros_like([5, 5]))
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nMag has the wrong shape but does not throw an error.")
        assert False

    def test_prepare_fieldmap_mask_wrong_shape(self):
        """Mask has the wrong shape."""

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine, mask=np.zeros_like([5, 5]))
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\nMask has the wrong shape but does not throw an error.")
        assert False

    def test_prepare_fieldmap_phasediff_1_echotime(self):
        """EchoTime of length one for phasediff should fail."""

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase], [self.echo_times[0]], self.affine)
        except RuntimeError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\necho_time has the wrong shape but does not throw an error.")
        assert False

    def test_prepare_fieldmap_3_echoes(self):
        """3 echoes are not implemented so the test should fail."""

        echo_times = [0.001, 0.002, 0.003]

        # This should return an error
        try:
            fieldmap = prepare_fieldmap([self.phase, self.phase, self.phase], echo_times, self.affine)
        except NotImplementedError:
            # If an exception occurs, this is the desired behaviour
            return 0

        # If there isn't an error, then there is a problem
        print("\n3 echoes are not implemented.")
        assert False

    def test_prepare_fieldmap_gaussian_filter(self):
        """ Test output of gaussian filter optional argument"""

<<<<<<< HEAD
        fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine, gaussian_filter=True)

        assert fieldmap.shape == self.phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        # assert np.all(np.isclose(fieldmap[30:35, 40, 0, 0],
        #                          np.array([18.51355514, 13.84794053,  9.48013154,  5.11232207,  0.64524454])))
=======
        fieldmap = prepare_fieldmap([self.phase], self.echo_times, self.affine, gaussian_filter=True, sigma=1)

        assert fieldmap.shape == self.phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        assert np.all(np.isclose(fieldmap[30:35, 40, 0, 0],
                                 np.array([19.46307638, 15.46251356, 11.05021768,  6.28096375,  1.30868717])))
>>>>>>> master
