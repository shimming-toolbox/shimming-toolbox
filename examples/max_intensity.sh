#!/usr/bin/env bash

################################################################################
# This script will calculate indices of EPI volumes that maximize signal
# intensity in the spinal cord for each slice. This is based on an EPI reference
# scan and was initially proposed in this article for z-shimming:
# https://onlinelibrary.wiley.com/doi/10.1002/hbm.26018.

# Input 1: Folder containing the dicoms.
# Input 2: Output folder. (Should be outside of input 1)

# Example: ./max_intensity.sh ~/path/to/dicom/folder ~/path/to/output/folder

# Hard requirement (Shim calculation): Shimming Toolbox: https://shimming-toolbox.org/en/latest
# Hard requirement (Spinal cord masking): SCT: https://spinalcordtoolbox.com/
# Soft requirements (Viewer): FSLeyes: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes
################################################################################

# Create an absolute path of the dicom folder
INPUT_PATH="$(cd "$(dirname "$1")" || exit; pwd)/$(basename "$1")"
# Check if the directory exists
if [ ! -d "${INPUT_PATH}" ]
then
    echo "Input path does not exist"
    exit
fi

# Create the absolute path of the output variable provided
OUTPUT_PATH="$(cd "$(dirname "$2")" || exit; pwd)/$(basename "$2")"
# Check if the output directory exists, if not, create it
if [ ! -d "${OUTPUT_PATH}" ]
then
    echo "Creating output folder"
    mkdir "${OUTPUT_PATH}"
fi

# Call dicom to nifti conversion and create a BIDS structure
st_dicom_to_nifti -i "${INPUT_PATH}" -o "${OUTPUT_PATH}" --subject "epi-shim" || exit

# Check if the derivatives directory exists, if not, create it
DERIV_ST_PATH="${OUTPUT_PATH}/derivatives/shimming-toolbox/sub-epi-shim"
if [ ! -d "${DERIV_ST_PATH}" ]
then
    echo "Creating Shimming Toolbox's derivatives folder"
    mkdir -p "${DERIV_ST_PATH}"
fi

# Calculate the mean
REF_EPI_PATH="${OUTPUT_PATH}/sub-epi-shim/func/sub-epi-shim_bold.nii.gz"
MEAN_PATH="${DERIV_ST_PATH}/sub-epi-shim_bold_mean.nii.gz"
st_maths mean --axis 3 -i "${REF_EPI_PATH}" -o "${MEAN_PATH}" || exit

# Check if the SCT derivatives directory exists, if not, create it
DERIV_SCT_PATH="${OUTPUT_PATH}/derivatives/sct/sub-epi-shim"
if [ ! -d "${DERIV_SCT_PATH}" ]
then
    echo "Creating SCT's derivatives folder"
    mkdir -p "${DERIV_SCT_PATH}"
fi

# Segment the spinal cord on the mean image
MASK_PATH="${DERIV_SCT_PATH}/ref_mean_mask.nii.gz"
sct_propseg -i "${MEAN_PATH}" -c "t2s" -radius "5" -o "${MASK_PATH}" || exit
# sct_propseg -i "${MEAN_PATH}" -c "t2s" -radius "5" -min-contrast "30" -o "${MASK_PATH}" || exit
# sct_propseg -i "${MEAN_PATH}" -c "t2s" -radius "5" -min-contrast "30" -max-deformation "5" -o "${MASK_PATH}" || exit

# Display the reference EPI and the mask
fsleyes "${REF_EPI_PATH}" "${MASK_PATH}" -cm red -a 70.0 &

# Launch the shim algorithm
OUTPUT_FILE="${DERIV_ST_PATH}/IntensitiesEPI.txt"
st_b0shim max-intensity -i "${REF_EPI_PATH}" --mask "${MASK_PATH}" -o "${OUTPUT_FILE}" -v "debug" || exit
