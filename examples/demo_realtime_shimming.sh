#!/usr/bin/env bash
#
# This function will generate static and dynamic (due to respiration) Gx, Gy, Gz components based on a field map time
# series (magnitude and phase images) and respiratory trace information obtained from Siemens bellows. An additional
# multi-gradient echo (MGRE) magnitude image is used to generate a ROI. The static and real-time components are
# extracted and decomposed into the optimal Gx, Gy and Gz values for each slice.

# Download example data
st_download_data testing_data

# Store testing_data path
TESTING_DATA_PATH="$(cd "$(dirname "testing_data")" || exit; pwd)/$(basename "testing_data")"

# Convert DICOMS to NIfTIs
st_dicom_to_nifti --input "${TESTING_DATA_PATH}/ds_b0/sub-realtime/sourcedata" --output "${TESTING_DATA_PATH}/ds_b0" --subject "realtime" || exit

# Create fieldmap
st_prepare_fieldmap "${TESTING_DATA_PATH}/ds_b0/sub-realtime/fmap/sub-realtime_phasediff.nii.gz" --mag "${TESTING_DATA_PATH}/ds_b0/sub-realtime/fmap/sub-realtime_magnitude1.nii.gz" --unwrapper "prelude" --output "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_fieldmap.nii.gz" --gaussian-filter True --sigma 1 || exit

# Mask target image
st_mask box --input "${TESTING_DATA_PATH}/ds_b0/sub-realtime/anat/sub-realtime_magnitude1.nii.gz" --size 15 15 20 --output "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_anat_mask.nii.gz" || exit

# Shim
st_b0shim realtime-dynamic --fmap "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_fieldmap.nii.gz" --target "${TESTING_DATA_PATH}/ds_b0/sub-realtime/anat/sub-realtime_magnitude1.nii.gz" --mask-static "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_anat_mask.nii.gz" --mask-riro "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_anat_mask.nii.gz" --resp "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/sub-realtime_PMUresp_signal.resp" --scanner-coil-order '1' --output-file-format-scanner "slicewise-hrd" --output "${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/realtime-shim" || exit

echo -e "\n\033[0;32mOutput is located here: ${TESTING_DATA_PATH}/ds_b0/derivatives/sub-realtime/realtime-shim"

# st_b0shim gradient_realtime will:
# - resample (in time) the physio trace to the 4d fieldmap data so that each time point of the fieldmap has its corresponding respiratory probe value.
# - Calculate voxelwise gradients for the fieldmap
# - Calculate a static offset component
# - Calculate a respiratory induce component (RIRO)
# - Output, riro, static and mean_p per slice in two text files that can be read by the sequence
