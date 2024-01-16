import os
import sys
import numpy as np
import twixtools
import numpy.linalg as npl
import nibabel as nib
import argparse
import json

# The script creates a mask for shimming single voxel MRS by converting MRS voxel position from scanner's coordinate to the image coordinate.
# This could be achieved with either passing XYZ coordiante directly in command line or by reading them from provided twix raw data.
# Mandatory rguments: -path: full source directory path, -fmap: filename of the fieldmap within the -path (should have the same name for both .nii.gz and .json files)

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description= "This script creates a mask to shim single voxel MRS. Scanner's XYZ coordiante and Voxel size can be directly given or be read from the raw-data")

    # Add command-line arguments
    parser.add_argument("-data_dir", metavar=':', type=str, help= "Source directory including twix data, fieldmap array and it's json fiel [mandatory]")
    parser.add_argument("-fmap", metavar=':', type=str, help= "filename of the acquired fieldmap to be shimmed without the extention [mandatory]")
    parser.add_argument("-raw", metavar=':', type=str, help= "filename of the twix raw data, without the extention [optional]")
    parser.add_argument("-X", metavar=':', type=float, help="scanner's X position in mm [optional]")
    parser.add_argument("-Y", metavar=':', type=float, help="scanner's Y position in mm [optional]")
    parser.add_argument("-Z", metavar=':', type=float, help="scanner's Z position in mm [optional]")
    parser.add_argument("-V", metavar=':', type=float, help= "MRS voxel size in mm [optional]")

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.data_dir is None or args.fmap is None:
        print("\033[91m"+"> Data directory and fieldmap filename should be provided, See: python auto_mask_MRS.py -h ")
        print(parser.description)
        sys.exit(1) # Exit the script with an error code


    # Check if any arguments were provided
    if (args.X) is not None and (args.Y) is not None and (args.Z) is not None and (args.V) is not None:
        # If X,Y,Z,V arguments were provided, execute this block
        X = args.X
        Y = args.Y
        Z = args.Z
        MRS_voxel_thick = args.V
        scanner_position = np.array([X, Y, Z, 1])
        data_dir = args.data_dir
        fmap = args.fmap
        print('scanner_position:', scanner_position)

    else:
        # If X,Y,Z,V arguments were NOT provided, execute this block
        if args.raw is None:
            print("\033[91m"+"> If X,Y,Z and V are not given directly the raw data must be given to read these info, See: python auto_mask_MRS.py -h")
            print(parser.description)
            sys.exit(1) # Exit the script with an error code
        else:
            print("Reading the twix raw data")
        raw_file_name = args.raw
        data_dir = args.data_dir
        fmap = args.fmap
        twix = twixtools.read_twix(os.path.join(data_dir, raw_file_name + '.dat'))

        # Access to the geometry of the voxel (all in mm).
        position_sag = (twix[-1]['hdr']['Spice']['VoiPositionSag'])
        position_cor = (twix[-1]['hdr']['Spice']['VoiPositionCor'])
        position_tra = (twix[-1]['hdr']['Spice']['VoiPositionTra'])
        MRS_voxel_thick = (twix[-1]['hdr']['Spice']['VoiThickness'])
        MRS_voxel_phaseFOV = (twix[-1]['hdr']['Spice']['VoiPhaseFOV'])
        MRS_voxel_readFOV = (twix[-1]['hdr']['Spice']['VoiReadoutFOV'])

        scanner_position= np.array([position_sag, -1* position_cor, position_tra, 1])
        print('scanner_position:', scanner_position)


    fmap_nii= nib.load(os.path.join(data_dir,fmap + '.nii.gz'))
    fmap_array= fmap_nii.get_fdata()
    fmap_affine= fmap_nii.affine
    print('reference_affine:', fmap_affine)
    print('reference_affine shape:', np.shape(fmap_affine))
    voxel_position= npl.inv(fmap_affine).dot(scanner_position)

    voxel_position = np.round(voxel_position)
    print('voxel_position:', voxel_position)

    I = voxel_position[0]
    J = voxel_position[1]
    K = voxel_position[2]

    # Open the JSON file
    json_file = os.path.join(data_dir,fmap+'.json')
    with open(json_file, 'r') as info:
        # Load the data from the file
        header = json.load(info)

    # Read the slice thickness of the reference GRE image.
    slice_thickness = header['SliceThickness'] # Isotropic fieldmap (mm)

    # As the given XYZ and hence IJK position is the voxel's center position, we need to define the voxel's dimension based on its thickness.
    SD = np.ceil(MRS_voxel_thick/(2*slice_thickness)) # Calculate the distance from center of MRS voxel to its edge based on number of fieldmap voxels.

    I, J, K, SD = int(I), int(J), int(K), int(SD)
    mask = np.zeros(fmap_array.shape) # create a zero mask with the same size as the input fieldmap to be shimmed.
    mask[I-SD:I+SD, J-SD:J+SD, K-SD:K+SD]=1 # Change the MRS voxel position to have 1 value.

    mask_nii = nib.Nifti1Image(mask, fmap_affine)
    nib.save(mask_nii,'mask_MRS.nii')
    print("\033[92m"+'MRS mask is created!')

if __name__ == "__main__":
    main()
