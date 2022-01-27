#!/usr/bin/env bash
#
# This function will generate static and dynamic (due to respiration) Gx, Gy, Gz components based on a fieldmap time
# series (magnitude and phase images) and respiratory trace information obtained from Siemens bellows. An additional
# multi-gradient echo (MGRE) magnitude image is used to generate an ROI and resample the static and dynamic Gx, Gy, Gz
# component maps to match the MGRE image. Lastly the average Gx, Gy, Gz values within the ROI are computed for each
# slice.
#
# The first variable should include the input path of the data to process

# Go inside input path
INPUT_PATH="$(cd "$(dirname "$1")" || exit; pwd)/$(basename "$1")"
cd "${INPUT_PATH}" || exit

# dcm2bids -d . -o rt_shim_nifti -p sub-example -c ../../config/dcm2bids.json
st_dicom_to_nifti --input "${INPUT_PATH}" --output "../rt_shim_nifti" --subject "sub-example" || exit
cd ../rt_shim_nifti/sub-example/fmap || exit

# Create fieldmap
st_prepare_fieldmap "sub-example_phasediff.nii.gz" --mag "sub-example_magnitude1.nii.gz" --unwrapper "prelude" --output "sub-example_fieldmap.nii.gz" || exit

# Mask anatomical image
mkdir "../../derivatives/sub-example"
st_mask box --input "../anat/sub-example_unshimmed_e1.nii.gz" --size 40 40 14 --output "../../derivatives/sub-example/sub-example_anat_mask.nii.gz" || exit

# Shim
st_b0shim gradient_realtime --fmap "sub-example_fieldmap.nii.gz" --anat "../anat/sub-example_unshimmed_e1.nii.gz" --resp "${INPUT_PATH}/PMUresp_signal.resp" --mask-static "../../derivatives/sub-example/sub-example_anat_mask.nii.gz" --mask-riro "../../derivatives/sub-example/sub-example_anat_mask.nii.gz" --output "../../derivatives/sub-example/gradient_realtime" || exit
st_b0shim realtime --fmap "sub-example_fieldmap.nii.gz" --anat "../anat/sub-example_unshimmed_e1.nii.gz" --mask-static "../../derivatives/sub-example/sub-example_anat_mask.nii.gz" --mask-riro "../../derivatives/sub-example/sub-example_anat_mask.nii.gz" --resp "${INPUT_PATH}/PMUresp_signal.resp" --scanner-coil-order '1' --output-file-format "slicewise-ch" --verbose "debug" --output "../../derivatives/sub-example/realtime" || exit

OUTPUT_PATH="$(dirname "${INPUT_PATH}")/rt_shim_nifti/derivatives/sub-example"
echo -e "\n\033[0;32mOutput is located here: ${OUTPUT_PATH}"

# st_b0shim gradient_realtime  will:
# - resample (in time) the physio trace to the 4d fieldmap data so that each time point of the fieldmap has its corresponding respiratory probe value.
# - Calculate voxelwise gradients for the fieldmap
# - Calculate a static offset component
# - Calculate a respiratory induce component (RIRO)
# - Output, riro, static and mean_p per slice in a text file that can be read by the sequence
