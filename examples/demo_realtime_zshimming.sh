#!/bin/bash
#
# This function will generate static and dynamic (due to respiration) Gz components based on a fieldmap time series
# (magnitude and phase images) and respiratory trace information obtained from Siemens bellows. An additional
# multi-gradient echo (MGRE) magnitude image is used to generate an ROI and resample the static and dynaminc Gz
# component maps to match the MGRE image. Lastly the average Gz values within the ROI are computed for each slice.

# Download example data
st_download_data testing_data

# Go inside folder
cd testing_data/realtime_zshimming_data || exit

# dcm2bids -d . -o nifti -p sub-example -c ../../config/dcm2bids.json
st_dicom_to_nifti -input . -output ../nifti -subject sub-example
cd ../nifti/sub-example/fmap || exit
# TODO: Name of phase2 should be phasediff

# Create fieldmap
st_prepare_fieldmap "sub-example_phasediff.nii.gz" -mag "sub-example_magnitude1.nii.gz" -unwrapper "prelude" -output "sub-example_fieldmap.nii.gz"

# Mask anatomical image
# Calling FSL directly
# TODO: st_mask
fslmaths sub-example_T2star_echo-1.nii.gz -thr 500 mask.nii.gz
fslmaths mask.nii.gz -bin mask.nii.gz
# Not implemented:
# <<
#st_mask -method sct
# Alternatively, you could run it with arbitrary shape:
# st_mask -method shape -shape cube -size 5 -o mask.nii
# >>

#TODO: st_realtime_zshim
st_realtime_zshim -fmap "sub-example_fieldmap.nii.gz" -anat "../anat/sub-example_unshimmed_e1.nii.gz" -resp "../../../PMUresp_signal.resp" -mask "TODO" 
# st_realtime_zshim will:
# - resample (in time) the physio trace to the 4d fieldmap data so that each time point of the fieldmap has its corresponding respiratory probe value.
# - Calculate voxelwise gradients for the fieldmap
# - Calculate a static offset component
# - Calculate a respiratory induce component (RIRO)
# - Output, riro, static and mean_p per slice in a text file that can be read by the sequence

