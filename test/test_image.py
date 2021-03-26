#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import nibabel as nib
import numpy as np

from shimmingtoolbox.image import concat_data
from shimmingtoolbox import __dir_testing__


class TestImageConcat(object):

    def setup(self):
        path_anat = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'anat')
        list_fname = [os.path.join(path_anat, 'sub-example_unshimmed_e1.nii.gz'),
                      os.path.join(path_anat, 'sub-example_unshimmed_e2.nii.gz'),
                      os.path.join(path_anat, 'sub-example_unshimmed_e3.nii.gz')]

        # Create nii list
        self.list_nii = []
        for i_file in range(len(list_fname)):
            nii_phase = nib.load(list_fname[i_file])
            self.list_nii.append(nii_phase)

    def test_concat_dim_4(self):
        out = concat_data(self.list_nii, 4)
        assert out.shape == self.list_nii[0].shape + (1, len(self.list_nii))  # (128, 68, 20, 1, 3)

    def test_concat_dim_1(self):
        out = concat_data(self.list_nii, 1)
        assert out.shape == (128, 204, 20)

    def test_concat_pixdim(self):
        pixdim = 0.2
        out = concat_data(self.list_nii, 2, pixdim=pixdim)
        assert np.all(np.isclose(out.header['pixdim'], [-1, 2.1875, 2.1875, 0.2, 1, 0, 0, 0]))

    def test_concat_100(self):
        """Use a list of more than 100 files"""
        out = concat_data(self.list_nii * 34, 4)
        assert out.shape == self.list_nii[0].shape + (1, len(self.list_nii) * 34)  # (128, 68, 20, 1, 3)
