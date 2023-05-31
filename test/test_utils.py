#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pathlib
import tempfile

from shimmingtoolbox.utils import create_output_dir
from shimmingtoolbox.utils import create_fname_from_path
from shimmingtoolbox.utils import is_similar_affine


def test_create_output_dir_folder():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        path_output = os.path.join(tmp, "new_folder")
        assert not os.path.exists(path_output)
        create_output_dir(path_output, is_file=False)
        assert os.path.exists(path_output)


def test_create_output_dir_file():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        path_output_folder = os.path.join(tmp, "new_folder")
        path_output = os.path.join(path_output_folder, "somefile.nii.gz")
        assert not os.path.exists(path_output_folder)
        create_output_dir(path_output, is_file=True)
        assert os.path.exists(path_output_folder)


def test_create_output_dir_folder_exists():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
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


def test_is_similar_affine():
    affine1 = np.array([[-2., 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.93],
                        [0., 0., 0., 1.]])

    affine2 = np.array([[-2., 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.90],
                        [0., 0., 0., 1.]])
    assert is_similar_affine(affine1, affine2)


def test_is_not_similar_affine():
    affine1 = np.array([[-2., 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.96],
                        [0., 0., 0., 1.]])

    affine2 = np.array([[-2., 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.90],
                        [0., 0., 0., 1.]])
    assert not is_similar_affine(affine1, affine2)


def test_is_not_similar_affine2():
    affine1 = np.array([[-2.0001, 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.96],
                        [0., 0., 0., 1.]])

    affine2 = np.array([[-2., 0., -0., 160.],
                        [-0., 2., -0., -108.],
                        [0., 0., 2., -103.96],
                        [0., 0., 0., 1.]])
    assert not is_similar_affine(affine1, affine2)
