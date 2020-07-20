#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti


def main():
    # Folder where the data will be downloaded
    data_path = 'data-testing-r20200713'

    # Download data when not already present
    if not os.path.isdir(data_path):
        DATA_URL = 'https://github.com/shimming-toolbox/data-testing/archive/r20200713.zip'
        fname_archive = 'data-testing.zip'
        print('Downloading test data from github')
        with urlopen(DATA_URL) as resp, open(fname_archive, 'wb') as out_file:
            shutil.copyfileobj(resp, out_file)
            with ZipFile(fname_archive, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall('')
        print('Data downloaded')

    unsorted_dicom_path = os.path.join(data_path, 'dicom_unsorted')

    # Create temporary folder for processing
    with tempfile.TemporaryDirectory() as tmp:
        # Path where the nifti files will be temporarily stored
        nifti_path = os.path.join(tmp, 'niftis')

        dicom_to_nifti(unsorted_dicom_path, nifti_path)
        print('Conversion from dicom to nifti done')
        breakpoint()

if __name__ == "__main__":
    main()
