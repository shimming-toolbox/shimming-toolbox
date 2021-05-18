Getting Started
===============

A good way to get started is looking at one of the `examples <https://github.com/shimming-toolbox/shimming-toolbox/tree/master/examples>`__ scripts.

demo_realtime_zshimming.sh
--------------------------

This  `shell script <https://github.com/shimming-toolbox/shimming-toolbox/blob/master/examples/demo_realtime_zshimming.sh>`__ calls different shimming-toolbox command lines functions to perform a whole shimming scenario. Acquisitions are downloaded from a `Github repository <https://github.com/shimming-toolbox/data-testing>`__ and output text files and quality control figures are generated.

More precisely, the script will calculate static and dynamic (due to
respiration) Gz components based on a fieldmap time series (magnitude and phase
images) and respiratory trace information obtained from Siemens bellows. An
additional multi-gradient echo (MGRE) magnitude image is used to generate an ROI
and resample the static and dynamic Gz component maps to match the MGRE image.
Lastly the average Gz values within the ROI are computed for each slice.
