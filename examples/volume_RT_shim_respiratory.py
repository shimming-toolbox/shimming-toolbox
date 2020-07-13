# Context: That scripts covers a volume-wise realtime shimming with respiratory probe scenario 
#
# Sort unorganized images from DICOM socket transfer
# dcm2bids is used to create folder structure and nifti conversion
# unsortedDicomDir: path to the folder containing the dicoms
# niftiPath: path where we want dcm2bids to store the nifti files
# 
# > dicom_to_nifti( unsortedDicomDir, niftiPath )
#
# Load images from the path
# Using niftiPath will prompt the user to select the appropriate acquisition. Alternatively, if the path to the acquisition is known, the path can be changed and the function won't prompt the user
# > mag = load_niftis( niftiPath ) 5D array (x, y, z, nEcho, nAcq)
# > phase = load_niftis( niftiPath ) 5D array (x, y, z, nEcho, nAcq)
#
# Convert to complex
# > complexArray = mag.*exp( 1i.*phase ); 5D array (x, y, z, nEcho, nAcq)
#
# Compute B0 fieldmap
# b0FieldMaps could be 2d, 3d or 4d. 4th dimension is the time in [s] (i.e. one B0 per timestamp).
# JCA: OK for s as unit?
# Note: the function should accommodate multiple fieldmap data (one per time point, in case of realtime shimming scenario)
# > unwrappedPhase = unwrap.unwrap_phase( complexArray, unwrapFunction )
# > b0FieldMaps = mapping( unwrappedPhase, echoTimes, mappingFunction )
#
# Load Siemens respiratory probe trace
# JCA: it would be nice to have the file "acdc45/probe02.resp" copied in this folder automatically, so everything pertaining to this experiment is at the same location.
# pmu is an object with useful properties.
# > pmu = load_probe_data(fullfile(pathNifti, 'probe02.resp'));
#
# Interpolate pmu to the image times
# JCA: Do we want the function below to output an object pmuInterp or do we want to keep the current behaviour: include pmu information inside "Field" object?
# pmuTrace is a vector
# > pmuTrace = match_probe_data_with_image(pmu, imgB0);
#
# Load T2w magnitude image for segmentation
# > imgAnat = load_niftis(fullfile(pathNifti, '04-T2w.nii'));
#
# Segment ROI
# > paramSeg.betParamDummy = XX  # some params for FSL BET function
# > paramSeg.sctDiameter = 30  # in mm
# > roi = segment_image(imgAnat, method={'bet', 'sct', 'cylinder'}, paramSeg)
#
# Optimize shim currents
# JCA: ShimOpt will use imgAnat as the destination target for regridding b0FieldMaps into the destination space. Also, if a segmentation exists in roi, it will use it to optimize shim currents.
# JCA: unclear to me "what" shims should be here (struct? object?).
# Nick's suggestion is to explicit all parameters as function input (instead of using a structure of parameters)
# another suggestion is to specify the algo type in the function itself
# Coil sensitivity WIP
# Output : (x y z channels)
# > coil = generate_coil(‘Nibardo’, [x y z])
#
# WIP
# > shims = optimize_shim_algo(b0FieldMaps, coil, method=‘realtimeZShim’, roi (optional), PMU (optional), etc.)
#
# Set shims to scanner or coil WIP
# > send_shim(shims, SyngoPath)
