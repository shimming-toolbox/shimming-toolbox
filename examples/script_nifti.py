from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti

def main():

    unsortedDicomDir = r"C:\Users\gaspa\Desktop\Test\dicom_unsorted"
    niftiPath = r"C:\Users\gaspa\Desktop\Test\nifti"
    dicom_to_nifti(unsortedDicomDir, niftiPath)

if __name__ == "__main__":
    main()