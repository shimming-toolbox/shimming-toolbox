#!/usr/bin/env bash
#
# This function will generate RF shim weights based on complex individual B1+ maps.

# Download example data
st_download_data testing_data

# Go inside folder
cd testing_data/ds_tb1 || exit

# Read B1+ NIfTI and compute RF shim weights for homogenization in the mask. Targets a value of 15nT/V.
st_b1shim --b1 "./sub-tb1tfl/rfmap/sub-tb1tfl_run-01_TB1map_uncombined.nii.gz" --mask "./derivatives/shimming-toolbox/sub-tb1tfl/sub-tb1tfl_mask.nii.gz" --algo 2 --target 15 --output ".." || exit
