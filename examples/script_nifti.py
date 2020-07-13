import os
import pathlib
import shutil
import sys
from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
import tempfile
import urllib.request as ur
from zipfile import ZipFile


def main():
    scriptPath = pathlib.Path(__file__).parent.absolute()  # Folder where this script is located
    dataPath = os.path.join(scriptPath, 'data_testing')  # Folder where the data will be downloaded

    # Download data when not already present
    if not os.path.isdir(dataPath):
        url = 'https://github.com/shimming-toolbox/data-testing/archive/r20200713.zip'
        try:
            fname_archive = 'data-testing.zip'
            with ur.urlopen(url) as resp, open(fname_archive, 'wb') as out_file:
                shutil.copyfileobj(resp, out_file)
                with ZipFile(fname_archive, 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall('.')
        except IndexError:
            print("ERROR - {0}:{1}".format(sys.exc_info()[0], sys.exc_info()[1]))

    # TODO: use systematic name for data-testing (could be in metadata of shimmingtoolbox)
    path_data = 'data-testing-r20200713'

    # Create temporary folder for processing
    tmp = tempfile.TemporaryDirectory()
    # Path where the niftis will be temporarily stored
    niftiPath = os.path.join(tmp.name, 'niftis')

    dicom_to_nifti(unsortedDicomDir, niftiPath)
    print('Conversion from dicom to nifti done')


if __name__ == "__main__":
    main()
