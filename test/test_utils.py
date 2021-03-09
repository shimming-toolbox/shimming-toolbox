import tempfile
import pathlib
import os
from shimmingtoolbox.utils import create_output_dir


def test_create_output_dir_folder():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_output = os.path.join(tmp, "new_folder")
        assert not os.path.exists(path_output)
        create_output_dir(path_output, is_file=False)
        assert os.path.exists(path_output)


def test_create_output_dir_file():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_output_folder = os.path.join(tmp, "new_folder")
        path_output = os.path.join(path_output_folder, "somefile.nii.gz")
        assert not os.path.exists(path_output_folder)
        create_output_dir(path_output, is_file=True)
        assert os.path.exists(path_output_folder)


def test_create_output_dir_folder_exists():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_output = tmp
        assert os.path.exists(path_output)
        create_output_dir(path_output, is_file=False)
        assert os.path.exists(path_output)
