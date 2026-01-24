import tempfile
import pathlib
import os

from shimmingtoolbox.utils import create_output_dir, create_fname_from_path, ms_past_midnight_to_iso_time

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


def test_create_fname_from_path():
    path = 'a/b/c'
    file = 'file.nii'

    fname = create_fname_from_path(path, file)

    assert fname == os.path.abspath("a/b/c/file.nii")


def test_create_fname_from_path_fname():
    path = 'a/b/c/file.nii'
    file = 'file2.nii'

    fname = create_fname_from_path(path, file)

    assert fname == os.path.abspath("a/b/c/file.nii")


def test_create_fname_from_path_2():
    path = '.'
    file = 'file.nii'

    fname = create_fname_from_path(path, file)

    assert fname == os.path.abspath("./file.nii")


def test_ms_past_midnight_to_iso_time():
    ms = 61923232.0
    iso = ms_past_midnight_to_iso_time(ms)
    assert iso == "171203.232000"
