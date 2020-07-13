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
            with ur.urlopen(url) as resp, open("osf_data.zip", 'wb') as out_file:
                shutil.copyfileobj(resp, out_file)
                with ZipFile("osf_data.zip", 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall(scriptPath)
        except IndexError:
            print("ERROR - {0}:{1}".format(sys.exc_info()[0], sys.exc_info()[1]))

    # TODO: use systematic name for data-testing (could be in metadata of shimmingtoolbox
    path_data = glob.glob('data-test*')[0]

    unsortedDicomDir = os.path.join(dataPath, 'dicom_unsorted')  # Path to the unsorted dicoms
    # Create temporary folder for processing
    tmp = tempfile.TemporaryDirectory()
    niftiPath = os.path.join(tmp.name, 'niftis')  # Path where the niftis will be temporarily stored

    dicom_to_nifti(unsortedDicomDir, niftiPath)
    print('Conversion from dicom to nifti done')


if __name__ == "__main__":
    main()
