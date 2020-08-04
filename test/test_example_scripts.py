# coding: utf-8

from pathlib import Path
import shutil
import os

from examples import general_demo
from shimmingtoolbox.utils import run_subprocess

class TestCore(object):

    def setup(self):
        # Get the directory where this current file is saved
        self.full_path = Path(__file__).resolve().parent
        self.path_nifti = os.path.join(self.full_path, 'niftis')
        self.path_testing_data = os.path.join(self.full_path, 'testing_data')
        self.fname_nifti = os.path.join(self.full_path, 'unwrap_phase_plot.png')

    def teardown(self):
        # Remove temporary files
        if os.path.isdir(self.path_nifti):
            shutil.rmtree(self.path_nifti)

        if os.path.isdir(self.path_testing_data):
            shutil.rmtree(self.path_testing_data)

        if os.path.isfile(self.fname_nifti):
            os.remove(self.fname_nifti)

    def test_demo_script_outputs_figure(self):

        general_demo.main()
        
        # Degug
        print(self.full_path)
        run_subprocess('ls {}'.format(self.full_path))

        assert (os.path.isfile(self.fname_nifti))
        assert (os.path.isdir(self.path_nifti))
        assert (os.path.isdir(self.path_testing_data))
