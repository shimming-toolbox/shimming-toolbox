# coding: utf-8

from pathlib import Path
import shutil
import os

from examples import general_demo


class TestCore(object):

    def setup(self):
        # Get the directory where this current file is saved
        self.test_path = Path(__file__).resolve().parent

        # Create tmp folder
        self.tmp_path = self.test_path / '__tmp__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()

    def teardown(self):
        # Remove temporary files
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)

    def test_demo_script_outputs_figure(self):

        fname_nifti = general_demo.main(self.test_path)

        assert (os.path.isfile(fname_nifti))

