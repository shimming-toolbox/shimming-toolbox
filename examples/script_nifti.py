#!/usr/bin/env python3

import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request as ur
from zipfile import ZipFile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti


def main():
    # Folder where this script is located
    script_path = pathlib.Path(__file__).parent.absolute()
    # Folder where the data will be downloaded
    data_path = os.path.join(script_path, 'data-testing-r20200713')

    # Download data when not already present
    if not os.path.isdir(data_path):
        url = 'https://github.com/shimming-toolbox/data-testing/archive/r20200713.zip'
        try:
            fname_archive = 'data-testing.zip'
            with ur.urlopen(url) as resp, open(fname_archive, 'wb') as out_file:
                shutil.copyfileobj(resp, out_file)
                with ZipFile(fname_archive, 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall(script_path)
            print('Downloading test data from github')
        except IndexError:
            print("ERROR - {0}:{1}".format(sys.exc_info()[0], sys.exc_info()[1]))

    unsorted_dicom_path = os.path.join(data_path, 'dicom_unsorted')
    # Create temporary folder for processing
    tmp = tempfile.TemporaryDirectory()
    # Path where the niftis will be temporarily stored
    nifti_path = os.path.join(tmp.name, 'niftis')

    dicom_to_nifti(unsorted_dicom_path, nifti_path)
    print('Conversion from dicom to nifti done')


if __name__ == "__main__":
    main()
