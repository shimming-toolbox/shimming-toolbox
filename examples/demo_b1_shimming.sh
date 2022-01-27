#!/usr/bin/env bash
#
# This function will generate RF shim weights based on complex individual B1+ maps.

# Download example data
st_download_data testing_data

# Go inside folder
cd testing_data/ds_b1 || exit

# Read B1+ NIfTI and compute RF shim weights for homogenization in the mask. Targets a value of 15nT/V.
st_b1shim --b1map "./sub-tb1tfl/fmap/sub-tb1tfl_TB1TFL_axial.nii.gz" --mask "./derivatives/shimming-toolbox/sub-tb1tfl/sub-tb1tfl_mask.nii.gz" --algo 2 --target 15 --output ".." || exit
