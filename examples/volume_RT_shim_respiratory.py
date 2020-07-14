# Context: That scripts covers a volume-wise realtime shimming with respiratory probe scenario 
#
# Sort unorganized images from DICOM socket transfer (this is how they are returned by the scanner)
# dcm2bids is used to convert the dicoms into niftis following the BIDS convension and organize them into a folder structure (separating magnitude and phase)
# unsorted_dicom_dir: path to the folder containing the unsorted dicoms
# nifti_path: path to the directory where we want dcm2bids to store the nifti files corresponding to the different acquisitions
# > dicom_to_nifti( unsorted_dicom_dir, nifti_path )
#
# Load nifti images from nifti_path
# Using nifti_path will prompt the user to select the appropriate acquisition (by displaying the different acquisitions stored in nifti_path)
# Alternatively, if the direct path to the acquisition is known, it can be directly use as an input and the function won't prompt the user
# mag & phase : 5D arrays (x y z nEcho nAcq)
# > mag = load_nii( nifti_path ) 
# > phase = load_nii( nifti_path )
# 
# Convert into a complex 5D array (x y z nEcho nAcq)
# > complex_array = np.multiply( mag, exp( phase*1j ) )
#
# Load T2w magnitude image for segmentation
# > img_anat = load_niftis(os.path.join(nifti_path, '04-T2w.nii'))
#
# Define the binary mask/ROI to shim
# Note: get_masking_image might call segment_spinal_canal (automatic) or define_mask (manual)
# masking_algo = defines what method will be used for ROI selection (manual selection, SCT, bet, cylinder...)
# > shim_VOI = get_mask( img_anat, masking_algo )
#
# Compute B0 fieldmaps
# b0_fieldmap could be 2d, 3d or 4d. The 4th dimension being the time in [s] (i.e. one B0 per timestamp).
# Note: the function should accommodate multiple fieldmap data (one per time point, in case of realtime shimming scenario)
# Note: map_field calls b0_mappers and might call unwrap_phase if needed
# threshold: sets the maximum frequency (Hz) value returned by map_field. If not defined, a default value is applied 
# > b0_fieldmap = map_field( complex_array, mapping_algo, unwrapping_algo, mask (optional), threshold (optional) )
#
# Load Siemens respiratory probe trace
# JCA: It would be nice to have the file "acdc45/probe02.resp" copied in this folder automatically, so everything pertaining to this experiment is at the same location.
# pmu_data: object with properties corresponding to the pressure measurements vs time
# > pmu_data = read_PMU( os.path.join( nifti_path, 'probe02.resp' ) )
#
# Interpolate pmu_data to the image times
# JCA: Do we want the function below to output an object pmu_interp or do we want to keep the current behaviour: include pmu information inside "Field" object?
# pmu_trace: vector
# Note: interp_PMU will call get_acq_times that will read the json files in the nifti_path to fetch the different acquisition times
# > pmu_trace = interp_PMU( pmu_data, nifti_path )
#
# Compute the respiration induced resonance offset
# > riro = get_riro( pmu_trace, b0_fieldmap )
#
# Returns the optimised field image that will be used for the shim coefficients computation
# > model_field( b0_fieldmap, riro )
#
# Coil sensitivity (WIP)
# Output : (x, y, z, channels)
# > coil = generate_coil(‘Nibardo’, [x y z])
#
# Optimize shim coefficients
# JCA: Shim_opt will use img_anat as the destination target for regridding b0_fieldmap into the destination space. Also, if a segmentation exists in roi, it will use it to optimize shim currents.
# Nick's suggestion is to explicit all parameters as function input (instead of using a structure of parameters). Another suggestion is to specify the algo type in the function itself
# Note: b0_fieldmap must be resliced to match img_anat's x, y and z dimensions.
# shims: need to define the variable type (matrix, text file...)
# > b0_resliced = reslice( b0_fieldmap, img_anat)
# > shims = shim_opt( b0_resliced, coil, method=‘realtimeZShim’, shim_VOI (optional), pmu_trace (optional), etc. )
#
# Send the shimming coefficient to scanner or coil (WIP)
# syngo_path: path to the folder where syngo reads the setting files
# > send_shim( shims, syngo_path )
