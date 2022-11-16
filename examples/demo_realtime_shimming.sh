#!/usr/bin/env bash
#
# This function will generate static and dynamic (due to respiration) Gx, Gy, Gz components based on a fieldmap time
# series (magnitude and phase images) and respiratory trace information obtained from Siemens bellows. An additional
# multi-gradient echo (MGRE) magnitude image is used to generate an ROI and resample the static and dynamic Gx, Gy, Gz
# component maps to match the MGRE image. Lastly the average Gx, Gy, Gz values within the ROI are computed for each
# slice.

# Download example data
st_download_data testing_data

# Store testing_data path
TESTING_DATA_PATH="$(cd "$(dirname "testing_data")" || exit; pwd)/$(basename "testing_data")"

# Go inside folder
cd testing_data/ds_b0/sub-realtime/sourcedata || exit

# dcm2bids -d . -o rt_shim_nifti -p sub-example -c ../../config/dcm2bids.json
st_dicom_to_nifti --input "." --output "../.." --subject "sub-gradient_realtime" || exit
cd ../../sub-gradient_realtime/fmap || exit

# Create fieldmap
st_prepare_fieldmap "sub-gradient_realtime_phasediff.nii.gz" --mag "sub-gradient_realtime_magnitude1.nii.gz" --unwrapper "prelude" --output "sub-gradient_realtime_fieldmap.nii.gz" --gaussian-filter True --sigma 1 || exit

# Mask anatomical image
mkdir "../../derivatives/sub-gradient_realtime"
st_mask box --input "../anat/sub-gradient_realtime_unshimmed_e1.nii.gz" --size 15 15 20 --output "../../derivatives/sub-gradient_realtime/sub-gradient_realtime_anat_mask.nii.gz" || exit

# Shim
st_b0shim gradient-realtime --fmap "sub-gradient_realtime_fieldmap.nii.gz" --anat "../anat/sub-gradient_realtime_unshimmed_e1.nii.gz" --resp "../../derivatives/sub-realtime/sub-realtime_PMUresp_signal.resp" --mask-static "../../derivatives/sub-gradient_realtime/sub-gradient_realtime_anat_mask.nii.gz" --mask-riro "../../derivatives/sub-gradient_realtime/sub-gradient_realtime_anat_mask.nii.gz" --output "../../derivatives/sub-gradient_realtime/gradient_realtime" || exit

echo -e "\n\033[0;32mOutput is located here: ${TESTING_DATA_PATH}/ds_b0/derivatives/sub-gradient_realtime/gradient_realtime"

# st_b0shim gradient_realtime will:
# - resample (in time) the physio trace to the 4d fieldmap data so that each time point of the fieldmap has its corresponding respiratory probe value.
# - Calculate voxelwise gradients for the fieldmap
# - Calculate a static offset component
# - Calculate a respiratory induce component (RIRO)
# - Output, riro, static and mean_p per slice in a text file that can be read by the sequence
