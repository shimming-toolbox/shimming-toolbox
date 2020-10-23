#!/bin/bash
#
# This shell script performs dual shimming for ASL application: one shim config
# for the labeling and another shim config for the imaging.

# Download example data
# TODO: include example ASL dataset
st_download_data testing_data

# Go inside folder
# TODO

# Convert dcm2nii
# TODO (if example dataset is DICOM)

# Compute fieldmap
# TODO

# Create masks: mask1 for ASL labeling region, mask2: imaging region
# TODO

# Generate a coil profile based on custom coils
# TODO

# Generate coil profile from SH basis
st_generate_profile_spherical_harmonics -i sub-example_fieldmap.nii.gz -order 2 -o coil_profile_sh2.nii.gz
st_generate_profile_spherical_harmonics -i sub-example_fieldmap.nii.gz -order 3 -o coil_profile_sh3.nii.gz

# Optimize shim in mask #1
# input:
# - mask
# - fieldmap
# - coil profile (could be multiple files)
# output:
# - Quality Control (QC): figures, log file (joblib)
# TODO

# Optimize shim in mask #2
# TODO

# Deal with outputs to be read by syngo console
# - txt file will be read by the pulse sequence
# - needs to be standardized across scenario (pulse seq)
# TODO

# Send txt file to syngo
# using mounted-drive, via ethernet socket
# TODO
