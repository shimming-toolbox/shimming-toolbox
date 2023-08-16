#!/usr/bin/python3
# -*- coding: utf-8 -*

import copy
import os
import pytest
import math
import nibabel as nib
import numpy as np

from shimmingtoolbox import __dir_testing__
from shimmingtoolbox.prepare_fieldmap import prepare_fieldmap
from shimmingtoolbox.prepare_fieldmap import correct_2pi_offset
from shimmingtoolbox.masking.threshold import threshold
from shimmingtoolbox.load_nifti import read_nii


class TestPrepareFieldmap(object):
    def setup(self):
        fname_phase = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.nii.gz')
        nii_phase = nib.load(fname_phase)
        phase = (nii_phase.get_fdata() * 2 * math.pi / 4095) - math.pi  # [-pi, pi]
        self.nii_phase = nib.Nifti1Image(phase, nii_phase.affine, header=nii_phase.header)
        fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
        nii_mag = nib.load(fname_mag)
        self.mag = nii_mag.get_fdata()
        self.echo_times = [0.00246, 0.00492]

    def test_prepare_fieldmap_1_echo(self):
        """Test default works."""
        fieldmap, _ = prepare_fieldmap([self.nii_phase], self.echo_times, self.mag, unwrapper='skimage')

        assert fieldmap.shape == self.nii_phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        assert np.allclose(fieldmap[30:35, 40, 0, 0],
                           np.array([18.5135551, 13.84794068, 9.48013143, 5.11232219, 0.64524455]))

    @pytest.mark.prelude
    def test_prepare_fieldmap_prelude(self):
        """Test prelude."""
        fieldmap, _ = prepare_fieldmap([self.nii_phase], self.echo_times, self.mag, unwrapper='prelude')

        assert fieldmap.shape == self.nii_phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        assert np.allclose(fieldmap[30:35, 40, 0, 0],
                           np.array([18.51407573, 13.85066883,  9.47872498,  5.11298149,  0.64801652]))

    def test_prepare_fieldmap_2_echoes(self):
        """Test 2 echoes works."""

        # Import 2 echoes and rescale
        fname_phase1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        nii_phase1 = nib.load(fname_phase1)
        phase1 = (nii_phase1.get_fdata() * 2 * math.pi / 4095) - math.pi
        nii_phase1_re = nib.Nifti1Image(phase1, nii_phase1.affine, header=nii_phase1.header)
        fname_phase2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        nii_phase2 = nib.load(fname_phase2)
        phase2 = (nii_phase2.get_fdata() * 2 * math.pi / 4095) - math.pi
        nii_phase2_re = nib.Nifti1Image(phase2, nii_phase2.affine, header=nii_phase2.header)

        # Load mag data to speed it prelude
        fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        mag = nib.load(fname_mag).get_fdata()

        echo_times = [0.0025, 0.0055]

        fieldmap, _ = prepare_fieldmap([nii_phase1_re, nii_phase2_re], echo_times, mag=mag, unwrapper='skimage')

        assert fieldmap.shape == phase1.shape

    # Tests that should throw errors
    def test_prepare_fieldmap_wrong_range(self):
        """Test error when range is not between -pi and pi."""
        nii = nib.Nifti1Image(self.nii_phase.get_fdata() - math.pi, self.nii_phase.affine, header=self.nii_phase.header)
        with pytest.raises(ValueError, match="Values must range from -pi to pi."):
            prepare_fieldmap([nii], self.echo_times, self.mag, unwrapper='skimage')

    def test_prepare_fieldmap_wrong_echo_times(self):
        """Wrong number of echo times."""

        echo_times = [0.001, 0.002, 0.003]
        with pytest.raises(ValueError, match="The number of echoes must match the number of echo times unless there is "
                                             "1 echo"):
            prepare_fieldmap([self.nii_phase], echo_times, self.mag, unwrapper='skimage')

    def test_prepare_fieldmap_mag_wrong_shape(self):
        """Mag has the wrong shape."""

        with pytest.raises(ValueError, match="mag and phase must have the same dimensions."):
            prepare_fieldmap([self.nii_phase], self.echo_times, mag=np.zeros_like([5, 5]), unwrapper='skimage')

    def test_prepare_fieldmap_mask_wrong_shape(self):
        """Mask has the wrong shape."""

        with pytest.raises(ValueError, match="Shape of mask and phase must match."):
            prepare_fieldmap([self.nii_phase], self.echo_times, self.mag, mask=np.zeros_like([5, 5]),
                             unwrapper='skimage')

    def test_prepare_fieldmap_phasediff_1_echotime(self):
        """EchoTime of length one for phasediff should fail."""

        with pytest.raises(ValueError, match="The number of echoes must match the number of echo times unless there is "
                                             "1 echo"):
            prepare_fieldmap([self.nii_phase], [self.echo_times[0]], self.mag, unwrapper='skimage')

    def test_prepare_fieldmap_3_echoes(self):
        """Test 3 echoes works."""

        fname_phase1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        fname_phase2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        mag = nib.load(fname_mag).get_fdata()
        nii1 = read_nii(fname_phase1)[0]
        nii2 = read_nii(fname_phase2)[0]
        data3 = nii1.get_fdata() + nii2.get_fdata()
        data3 = np.angle(np.exp(1j * data3))
        nii3 = nib.Nifti1Image(data3, nii1.affine, header=nii1.header)
        echo_times = [0.0025, 0.0055, 0.008]

        fieldmap, _ = prepare_fieldmap([nii1, nii2, nii3], echo_times, mag, unwrapper='skimage')
        assert fieldmap.shape == nii1.shape

    def test_prepare_fieldmap_3_echoes_timeseries(self):
        fname_phase1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        fname_phase2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_magnitude1.nii.gz')
        nii_mag = nib.load(fname_mag)
        data_mag = np.repeat(nii_mag.get_fdata()[..., np.newaxis], 3, 3)

        nii1 = read_nii(fname_phase1)[0]
        data1 = np.repeat(nii1.get_fdata()[..., np.newaxis], 3, 3)
        nii1_t = nib.Nifti1Image(data1, nii1.affine, header=nii1.header)

        nii2 = read_nii(fname_phase2)[0]
        data2 = np.repeat(nii2.get_fdata()[..., np.newaxis], 3, 3)
        nii2_t = nib.Nifti1Image(data2, nii2.affine, header=nii2.header)

        data3 = nii1_t.get_fdata() + nii2_t.get_fdata()
        data3 = np.angle(np.exp(1j * data3))
        nii3_t = nib.Nifti1Image(data3, nii1.affine, header=nii1.header)
        echo_times = [0.0025, 0.0055, 0.008]

        fieldmap, _ = prepare_fieldmap([nii1_t, nii2_t, nii3_t], echo_times, data_mag, unwrapper='skimage')
        assert fieldmap.shape == nii1_t.shape

    def test_prepare_fieldmap_wrong_threshold(self):
        """EchoTime of length one for phasediff should fail."""

        with pytest.raises(ValueError, match="Threshold should range from 0 to 1. Input value was:"):
            prepare_fieldmap([self.nii_phase], self.echo_times, self.mag, threshold=2, unwrapper='skimage')

    def test_prepare_fieldmap_gaussian_filter(self):
        """Test output of gaussian filter optional argument"""

        fieldmap, _ = prepare_fieldmap([self.nii_phase], self.echo_times, self.mag, gaussian_filter=True, sigma=1,
                                       unwrapper='skimage')

        assert fieldmap.shape == self.nii_phase.shape
        # If the behaviour of the called function is modified, this assertion below should capture it:
        assert np.all(np.isclose(fieldmap[30:35, 40, 0, 0],
                                 np.array([18.46090157, 14.46425859, 10.1252873, 5.46828194, 0.56341495])))


def test_correct_2pi_offset():
    """Test if it corrects a 2pi offset"""
    fname_mag = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_magnitude1.nii.gz')
    mag = nib.load(fname_mag).get_fdata()
    fmap = np.zeros_like(mag)

    # Add 2pi to one timepoint
    mask = threshold(mag)
    fmap_rad_with_offset = copy.deepcopy(fmap)
    fmap_rad_with_offset[..., 2] += mask[..., 2] * 2 * np.pi

    fmap_corrected = correct_2pi_offset(fmap, mag, mask, validity_threshold=0.2)

    assert np.all(np.isclose(fmap_corrected, fmap))
