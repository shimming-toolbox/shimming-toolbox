#!/bin/bash
#
# This function will generate static and dynamic (due to respiration) Gz components based on a fieldmap time series
# (magnitude and phase images) and respiratory trace information obtained from Siemens bellows. An additional
# multi-gradient echo (MGRE) magnitude image is used to generate an ROI and resample the static and dynaminc Gz
# component maps to match the MGRE image. Lastly the average Gz values within the ROI are computed for each slice.

# Download example data
st_download_data testing_data

# Go inside folder
cd testing_data/realtime_zshimming_data

st_dicom_to_nifti -i . -o nifti -sub sub-example
cd nifti/sub-example
cd fmap

st_unwrap_phase -i XX -method prelude
st_compute_b0field -i XX -o fieldmap.nii
# fieldmap.nii is a 4d file, with the 4th dimension being the time. Ie: one B0 field per time point.

st_mask -method sct
# Alternatively, you could run it with arbitrary shape:
# st_mask -method shape -shape cube -size 5 -o mask.nii

# Use the provided coil profile (download them first-- could be done during installation of shimming-toolbox)
st_download_data coil_profiles
# Alternatively: generate a coil profile using custom coil geomtry info
#st_generate_coil_profile -i coil_geometry_in_i_dunno_what_format.?? -o my_coil_profile.nii

st_shim -fmap fieldmap.nii -coil-profile $SHIM_DIR/coils/siemens_terra.nii -mask mask.nii -physio XX -method {volumewise, slicewise}
# st_shim will:
# - resample coil profile into the space of fieldmap.nii
# - resample (in time) the physio trace to the 4d fieldmap data so that each time point of the fieldmap has its corresponding respiratory probe value.
# - run optimizer within mask.nii
# - outputs:
#   - fieldmap_shimmed.nii
#   - coefficients.csv
#   - figures

# Output text file to be read by syngo console
# TODO