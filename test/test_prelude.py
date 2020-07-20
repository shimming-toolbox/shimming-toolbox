# coding: utf-8

from pathlib import Path
import shutil
import requests
from zipfile import ZipFile
import os
import glob
import nibabel as nib
import numpy as np

import pytest

from shimmingtoolbox.unwrap import prelude


class TestCore(object):

    def setup(self):
        # Get the directory where this current file is saved
        self.full_path = Path(__file__).resolve().parent
        print(self.full_path)
        # "test/" directory
        self.test_path = self.full_path

        # Create temp folder
        # Get the directory where this current file is saved
        self.tmp_path = self.test_path / '__tmp__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()
        print(self.tmp_path)

        self.download_data()

    def teardown(self):
        # Get the directory where this current file is saved
        print(self.full_path)
        print(self.test_path)
        print(self.test_path)

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
            pass

    def download_data(self):
        # Download example data
        url = 'https://github.com/shimming-toolbox/data-testing/archive/r20200709.zip'
        filename = os.path.join(self.tmp_path, 'data-testing.zip')

        r = requests.get(url)
        open(filename, 'wb').write(r.content)

        with ZipFile(filename, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=self.tmp_path)
        os.remove(filename)
        # TODO: use systematic name for data-testing (could be in metadata of shimmingtoolbox

    @pytest.mark.unit
    def test_prelude_default_works(self):
        path_data = glob.glob(os.path.join(self.tmp_path, 'data-test*'))[0]

        # Open phase data
        fname_phases = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*phase*.nii.gz'))
        if len(fname_phases) > 2:
            raise IndexError('Phase data parsing is wrongly parsed')

        nii_phase_e1 = nib.load(fname_phases[0])
        nii_phase_e2 = nib.load(fname_phases[1])

        # Scale to phase to radians
        phase_e1 = nii_phase_e1.get_fdata() / 4096 * 2 * np.pi - np.pi
        phase_e2 = nii_phase_e2.get_fdata() / 4096 * 2 * np.pi - np.pi

        # Open mag data
        fname_mags = glob.glob(os.path.join(path_data, 'sub-fieldmap', 'fmap', '*magnitude*.nii.gz'))

        if len(fname_mags) > 2:
            raise IndexError('Mag data parsing is wrongly parsed')

        nii_mag_e1 = nib.load(fname_mags[0])
        nii_mag_e2 = nib.load(fname_mags[1])

        unwrapped_phase_e1 = prelude(phase_e1, nii_mag_e1.get_fdata(), nii_phase_e1.affine)
        unwrapped_phase_e2 = prelude(phase_e2, nii_mag_e2.get_fdata(), nii_phase_e1.affine)

        # Compute phase difference
        unwrapped_phase = unwrapped_phase_e2 - unwrapped_phase_e1

        # Compute wrapped phase diff
        wrapped_phase = phase_e2 - phase_e1

        assert(unwrapped_phase.shape == wrapped_phase.shape)

# TODO: More thorough tests. (eg: Test for all different options, test error handling, etc)
