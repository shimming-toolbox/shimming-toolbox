# Changelog

## 1.0 (2024-08-20)

### PACKAGE: SHIMMING TOOLBOX

**FEATURE**
 - **st_b0shim**: Add shim CLI and sequencer. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/296)
 - **st_b0shim**: Implement order 2 in st_b0shim static and realtime CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/340)
 - **st_b0shim**: Deal with saturation pulse when B0 shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/374)
 - **st_b0shim**: Add standard deviation criteria and options to select it in st_b0shim CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/405)
 - **st_b0shim**: Implement maximization of shim settings based on signal intensity. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/413)
 - **st_b0shim**: Faster B0 shimming by changing some mathematical expressions. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/432)
 - **st_b0shim**: Add compatibly for Philips scanners. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/435)
 - **st_b0shim**: Changed the optimizer to a quadratic problem. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/458)
 - **st_b0shim**: Add GE compatibility, spherical harmonic shim + refactoring of the Scanner shim coordinate systems. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/476)
 - **st_b1shim**: Added argument for VOP path in b1shim cli. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/315)
 - **st_b1shim**: Improve B1 shimming: implement resampling of masks into B1 maps + phase only shimming and add documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/321)
 - **st_check_dependencies, st_mask**: first iteration to adding BET. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/542)
 - **st_create_coil_profiles**: Implement coil profile generation API/CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/312)
 - **st_create_coil_profiles**: Coil profiles creation from CAD geometries. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/467)
 - **st_create_coil_profiles**: Add CLI and GUI command to create constraint files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/547)
 - **st_mask**: Add a cli to create a spherical mask. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/425)
 - **st_prepare_fieldmap**: Use --savemask option from prelude to calculate the output mask in st_prepare_fieldmap. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/393)
 - **st_prepare_fieldmap**: Multi echo field mapping. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/419)
 - **st_prepare_fieldmap**: Implement skimage's phase unwrapper. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/473)
 - **st_realtime_shim**: Add realtime sequencing ability. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/280)
 - **st_sort_dicoms**: Add a CLI to sort DICOMs in separate folders. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/428)
 - **st_unwrap**: Add unwrap CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/520)

**ENHANCEMENT**
 - **st_b0shim**: Add gradient shimming output file format to the b0 static CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/339)
 - **st_b0shim**: Resample input masks on the target anat when B0 shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/376)
 - **st_b0shim**: Parse slice ordering from BIDS json sidecar. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/378)
 - **st_b0shim**: Reduce memory usage and speed things up when shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/383)
 - **st_b0shim**: Add fatsat option in dynamic and realtime dynamic b0shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/384)
 - **st_b0shim**: Allow 4d masks and target anatomical when using dynamic shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/389)
 - **st_b0shim**: Use multiprocessing's fork method to speed up optimizing. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/390)
 - **st_b0shim**: Create output directory automatically for st_b0shim max-intensity. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/418)
 - **st_b0shim**: Log the indexes when using st_b0shim max-intensity. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/420)
 - **st_b0shim**: Faster B0 shimming . [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/423)
 - **st_b0shim**: Add ability to shim using 2D field maps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/431)
 - **st_b0shim**: Make Resampling faster by using multiprocessing with joblib. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/443)
 - **st_b0shim**: Changed the optimization to get faster results. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/451)
 - **st_b0shim**: Improve real-time shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/462)
 - **st_b0shim**: Shimmed field map plotting for real time shimming by averaging over time dimension. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/463)
 - **st_b0shim**: Split coils and scanner order for optimization + allow specific scanner orders for each. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/484)
 - **st_b0shim**: Add 3rd order spherical harmonics. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/497)
 - **st_b0shim**: Implement Philips+GE related spherical harmonics conventions. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/498)
 - **st_b0shim, st_image**: Improve output of st_b0shim and add st_image logical-and. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/335)
 - **st_b1shim**: Implement RF shimming code. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/299)
 - **st_b1shim**: Improved RF shimming output figure. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/316)
 - **st_b1shim**: Add norm constraint to shim-weights when not SAR constrained. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/318)
 - **st_b1shim**: Minor improvements of B1+ shimming functionalities. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/333)
 - **st_b1shim**: Add regularization term to CV reduction. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/352)
 - **st_b1shim**: Real/imaginary B1+ shim weights splitting instead of magnitude/phase. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/354)
 - **st_b1shim**: Add output B1 shim weights as a machine friendly text file. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/457)
 - **st_b1shim, st_dicom_to_nifti**: Fix B1+ maps right after their conversion into NIfTI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/344)
 - **st_create_coil_profiles**: Add ability to input a mask when creating coil profiles. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/546)
 - **st_dicom_to_nifti**: Add output of phase for anat images in dcm2bids. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/309)
 - **st_dicom_to_nifti**: Improve dcm2bids config file and fieldmap conversion. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/386)
 - **st_dicom_to_nifti**: Add verbose to the st_dicom_to_nifti. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/416)
 - **st_dicom_to_nifti**: Add mp2rage, and more field maps to dcm2bids config file and update to version >= 2.1.7. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/466)
 - **st_dicom_to_nifti**: New dcm2bids config file. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/500)
 - **st_download_data**: Update link to the newest version of the coil profile tutorial. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/472)
 - **st_image**: Resample in logical-and if the volumes are not the same orientation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/396)
 - **st_mask**: Update nibabel to latest version and allow to input a scaled threshold. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/477)
 - **st_prepare_fieldmap**: Update prepare_fieldmap CLI/API to accept non siemens data more easily. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/294)
 - **st_prepare_fieldmap**: Correct for possible 2*pi offsets across time points when unwrapping. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/330)
 - **st_prepare_fieldmap**: Add option to save the calculated mask when using st_prepare_fieldmap. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/391)
 - **st_sort_dicoms**: Allow to sort any DICOM files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/459)
 - **st_sort_dicoms**: Add recursive option to sort dicoms. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/492)
 - Handling of different slice orientation when reading b1 maps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/283)
 - Added handling of coronal RF nifti + tests for all orientations. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/308)
 - Allow spaces in the CLIs. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/387)
 - Changed version of matplotlib. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/454)
 - Add ability to simulate x, y and z gradients. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/485)

**BUG**
 - **st_b0shim**: Fix bug where there were dead voxels in the calculated shim figures. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/394)
 - **st_b0shim**: Improve mask resampling. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/395)
 - **st_b0shim**: Invert polarity of the frequency adjust. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/440)
 - **st_b0shim**: Fixed a particular case, when there is nothing to shimmed in an optimization. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/442)
 - **st_b0shim**: Solved a bug where mask_fmap and the dilated mask where switched. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/460)
 - **st_b0shim**: Fix an issue where real-time shimming would not work using pseudo-inverse. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/465)
 - **st_b0shim**: Resolve quadrog issue when the sum of the channels constraint was infinite. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/480)
 - **st_b0shim**: Bug fix: "absolute" output format. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/510)
 - **st_b0shim**: Update custom coil config file. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/512)
 - **st_b0shim**: Update how constraints are handled. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/530)
 - **st_b0shim**: Change erode to dilate for optimization. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/538)
 - **st_b0shim**: Fix a bug when scanner constraints are not implemented. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/543)
 - **st_b0shim, st_mask, st_prepare_fieldmap**: Fix st_b0shim realtime-dynamic order 0 and output folder for masking and fieldmap. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/353)
 - **st_check_dependencies**: Change command that checks for sct installation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/298)
 - **st_check_dependencies**: Change SCT CLI call to check version. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/402)
 - **st_dicom_to_nifti**: Fix phasediff renaming issue for dual-echo fieldmaps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/447)
 - **st_dicom_to_nifti**: Update dcm2bids to version 3.0.1. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/470)
 - **st_dicom_to_nifti**: Improve reliability of phase diff renaming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/504)
 - **st_dicom_to_nifti**: Solve dcm2niix not in PATH issue and improve st_prepare_fieldmap documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/523)
 - **st_dicom_to_nifti, st_prepare_fieldmap**: Bug fix related to updating FSLeyes version and Shimming Toolbox version. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/483)
 - **st_download_data**: Fix broken link in st_download_data data_create_coil_profiles. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/535)
 - **st_prepare_fieldmap**: Field mapping pipeline improvement. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/410)
 - **st_prepare_fieldmap**: Resolve prelude error if --savemask directory is not already created when field mapping. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/448)
 - **st_prepare_fieldmap**: Change phase range in field mapping calculations. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/499)
 - **st_realtime_shim**: Fix realtime shimming mask resampling modes. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/289)
 - **st_realtime_shim**: gradient realtime shim output coordinate system. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/310)
 - **st_realtime_shim**: Gradient realtime_shim orientation and nan problems. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/323)
 - Solve launchers needing to be launched from the ST venv. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/346)
 - Restrict numpy version < 2.x. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/527)

**INSTALLATION**
 - Automate the install process. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/295)
 - Remove the need for sudo in the installer. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/304)
 - Update pillow and numpy due to Github security alert. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/328)
 - Resolve deprecated warnings for np.float. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/337)
 - Update required importlib-metadata version. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/362)
 - Share a single conda environment.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/379)
 - Switch to using mambaforge. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/403)
 - Add windows compatibility. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/469)

**REFACTORING**
 - **st_b0shim**: Improve dynamic B0 shimming workflow. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/388)
 - **st_b0shim**: Refactoring of the B0 shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/439)
 - **st_b1shim**: Reorganizeb1 shimming files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/314)
 - **st_dicom_to_nifti**: Update st_dicom_to_nifti help. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/338)
 - **st_prepare_fieldmap**: Updates fieldmapping APIs/CLIs to output proper NIfTI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/305)
 - Adapt CI tests to updated testing data organization. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/319)
 - Replace remaining paths by fname. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/345)

**TESTING**
 - **st_b1shim**: Add unit test for unknown RF shimming algorithm. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/317)
 - Add missing prelude test marker. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/399)

**DOCUMENTATION**
 - **st_b1shim**: Specify in docstring that CV reduction favors high B1+ efficiency. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/357)

### PACKAGE: PLUGIN

 **FEATURE**
  - **st_check_dependencies, st_mask**: first iteration to adding BET. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/542)
  - **st_create_coil_profiles**: Add CLI and GUI command to create constraint files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/547)

 **INSTALLATION**
  - Update to the newest version of Shimming Toolbox [9c454d7]. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/24)
  - Update ST installation. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/32)
  - Update wxpython to the latest version. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/35)
  - Use mamba. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/36)
  - Share a single conda environment.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/379)
  - Share a single conda environment.. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/37)
  - Bump version and implement features. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/52)
  - Bump version to include coil profile generation. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/55)
  - Update version of ST. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/56)
  - Bump version to include dual-echo field mapping fix. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/57)
  - Change the version of fsleyes and bump the Shimming Toolbox version. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/63)
  - Bump Shimming Toolbox version . [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/66)
  - Update fsleyes to 1.12.4. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/71)


 **ENHANCEMENT**
  - Update installer to integrate with new installer in `shimming-toolbox`. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/19)
  - Gc/b1. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/22)
  - Fetch CLI docstrings. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/31)
  - Bump version, implement relevant features and bug fixes. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/39)
  - Add realtime output of CLIs to fsleyes terminal. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/41)
  - Implement maximization of shim settings based on signal intensity in the plugin. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/51)
  - Auto load ST on startup and version bump. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/64)
  - Adapt GUI to new split optimization. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/65)
  - Add 3rd order support. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/68)
  - Ab/signal recovery. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/70)

 **BUG**
  - Minor bug fixes. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/27)
  - Improve installation. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/29)
  - Solve bug preventing to launch Shimming Toolbox on linux. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/54)
  - Allow to select files from the file dialog box and nested dropdowns. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/58)
  - bug fix. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/67)

 **DOCUMENTATION**
  - Add issue template. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/16)
  - Add pull request template. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/18)
  - Add LICENSE. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/62)

 **TESTING**
  - Add continuous integration and pre-commit. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/61)

 **REFACTORING**
  - Gc/cleanup. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/40)
  - Major refactoring. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/59)
  - Merge fsleyes-plugin-shimming-toolbox repo in Shimming Toolbox. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/533)





### REPO

**ENHANCEMENT**
 - Add GRE-T1w to config file. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/531)

**BUG**
 - Update version of read the docs theme. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/468)
 - Use python 3.10 for windows. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/522)

**DOCUMENTATION**
 - add badges to readme for the license and release. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/284)
 - Improve installation instructions. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/306)
 - Implement GUI example scenario. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/347)
 - Simplify installation documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/348)
 - Add GUI example tuto for b1 shim. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/366)
 - Update installation documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/367)
 - Add Siemens compatibility warning to B1+ tutorial. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/372)
 - Update names of subcommand in the documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/417)
 - Add information about citation to the documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/421)
 - Removed unecessary explanations. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/422)
 - Update b1_shimming.rst. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/456)
 - Updated documentation to encourage people to post on the forum. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/516)

**INSTALLATION**
 - Resolve deprecated st_venv instructions and calls. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/400)
 - Add windows compatibility. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/469)
 - Upgrade to python 3.10. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/508)

**TESTING**
 - removed ubuntu 16 from test.yml. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/301)
 - Fix hanging tests in CI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/331)
 - Version/python version updates for pre-commit, sklearn and read the docs. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/415)
 - Change ubuntu 18.04 to 22.04. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/445)
 - Removed macos-10.15 for 11.0 and 12.0. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/449)
 - Update GitHub actions to V3. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/453)
 - Remove macos-11 and add macos-13 in CI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/503)

## 0.1 (2021-07-28)

### PACKAGE: SHIMMING TOOLBOX

**BUG**
 - **st_check_dependencies**: Refactored prelude and dcm2niix to avoid bug when not on PATH. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/275)
 - **st_dicom_to_nifti**: fix remove_tmp argument in dicom_to_nifti. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/132)
 - Repair dicom_to_nifti missing calls [bug]. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/78)
 - Update generate meshgrid. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/168)
 - Fix read_nii not looking for ImageComments and ImageType. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/171)
 - Fix singleton on last dimension error in prelude. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/175)
 - Fix local error with load_nifti. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/179)

**FEATURE**
 - **st_check_dependencies**: Check installation and versions of 3rd party binaries. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/176)
 - **st_download_data**: Added functions to download data. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/25)
 - **st_image, st_maths**: Add ability to average across files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/250)
 - **st_mask**: Add square, cube and threshold cli. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/137)
 - **st_mask**: Create CLI for SCT mask. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/249)
 - **st_prepare_fieldmap**: Add ability to create fieldmaps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/169)
 - **st_prepare_fieldmap**: eao/gaussian-filter. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/200)
 - **st_realtime_shim**: Realtime_zshim shim CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/134)
 - **st_realtime_shim**: Output in-plane (x and y) shim coefficients. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/270)
 - Create numerical model data for multi-echo field maps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/5)
 - Add unwrap phase functionality. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/15)
 - Implement dicom_to_nifti: wrapper for dcm2bids to convert DICOM data. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/18)
 - Add the ability to load niftis. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/35)
 - Added masking functionality. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/60)
 - Implementing basic optimizer and test bench. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/73)
 - Create synthetic coil-profiles. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/74)
 - Added API for dealing with coordinate systems. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/135)
 - Add ability to read from the PMU. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/142)
 - Add ability to get timing data from scans. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/149)
 - Updated load_nifti function to handle rf-maps obtained with Turboflash B1 mapping . [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/235)
 - Support fsleyes as a GUI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/248)

**ENHANCEMENT**
 - **st_dicom_to_nifti**: Update dicom_to_nifti with a CLI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/120)
 - **st_dicom_to_nifti**: Rename phasediff image in dicom_to_nifti. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/191)
 - **st_dicom_to_nifti**: Added possibility to identify TFL sequence for rf mapping. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/227)
 - **st_dicom_to_nifti**: Update dcm2bids to handle all TFL B1 maps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/247)
 - **st_mask**: Implementation of a test to verify the correct creation of the mask. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/197)
 - **st_prepare_fieldmap**: Update prepare fieldmap . [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/189)
 - **st_realtime_shim**: Add RIRO and Static masks to realtime zshimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/221)
 - **st_realtime_shim**: optimization by avoiding loops for regression over numpy array.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/244)
 - Add a shimming scenario script under /examples. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/9)
 - Updated general demo script: general_demo.py. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/46)
 - Added example shell script for realtime z-shimming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/92)
 - Optimizing optimizer code to run faster on large fieldmaps with small masks. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/124)
 - Optimizer improvements. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/129)
 - Fixed biot-savart scaling. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/141)
 - Compute gradient in physical and voxel coordinate systems. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/181)
 - Raise warning when loading RF nifti with missing SliceTiming. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/252)
 - Refactor and add features to Optimizers. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/253)

**TESTING**
 - **st_download_data**: Update download_data CLI dataset links. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/106)
 - **st_download_data**: Updated URL for download data testing. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/111)
 - **st_download_data**: Update download data link. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/195)
 - **st_download_data**: Updated the testing data release version. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/214)
 - **st_download_data**: Update download_data link to the latest release. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/226)
 - Update __dir_testing__ to be an absolute path. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/131)
 - Ensure test environment is set up with test data and soft dependencies before running tests. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/136)
 - Updates link for testing-data. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/138)

**REFACTORING**
 - [chore][cli/download_data] Remove stale TODO. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/90)
 - Remove unneeded example modules. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/153)
 - Change prelude mag parameter to be an optional argument. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/178)

**INSTALLATION**
 - Add basic python package setup. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/4)
 - packaging: only install tests when asked. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/7)
 - Upgrade dcm2niix to latest.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/43)
 - Moved testing and docs packages in default installation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/97)
 - Update minimum version to Python 3.7. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/103)
 - Update shimming-toolbox package install. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/146)

**DOCUMENTATION**
 - Sketch for some command line tools in the toolbox.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/6)
 - Source package version from setup.py.. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/19)
 - [docs][fix] Add documentation dependency installation steps. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/62)
 - Docstring fixes to address Sphinx build warnings. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/72)

### PACKAGE: PLUGIN

**FEATURE**
 - Support fsleyes as a GUI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/248)

 **REFACTORING**
  - Convert the plugin into a fsleyes package. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/2)

 **FEATURE**
  - Support fsleyes as a GUI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/248)

  **ENHANCEMENT**
   - Add GUI Integration into FSLeyes. [View pull request](https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox/pull/1)

### REPO

**DOCUMENTATION**
 - Create CONTRIBUTING.rst and LICENSE. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/40)
 - Port shimming-toolbox documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/41)
 - [docs][README] Add RTD build sticker. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/49)
 - [docs][API reference] Add unwrap module to API reference. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/56)
 - [docs][API reference] Replace Autodoc with Napoleon. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/57)
 - [fix][download_data] Automatically generate dataset info in CLI help message. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/93)
 - Fix link to installation instructions. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/104)
 - Satisfy GitHub community profile checklist. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/107)
 - Fixed former repos name. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/108)
 - Update repo name in documentation and project config files. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/115)
 - [docs][installation] Add FSL Prelude as dependency. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/126)
 - Fix RTD badge. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/147)
 - Add documentation stub for dicom_to_nifti.py. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/166)
 - Create new CLI documentation section. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/185)
 - Fix GH Actions badge in README. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/188)
 - Revised index.rst and leading numbers on RTD directories. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/201)
 - Add the Twitter badge. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/203)
 - Update links in README.md. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/213)
 - Update docs and readme for new dcm2niix version. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/231)
 - Move images to new repo. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/255)
 - Update the documentation with up to date info and figures. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/256)
 - Update documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/259)
 - Reordered listed 3rd party software, added FSLeyes. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/261)
 - Added FSLeyes example figure. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/265)
 - Addition of the commands accessible in shimming-toolbox in the README.rst. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/276)
 - Update API and CLI documentation. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/277)

**TESTING**
 - Install dcm2niix in CI. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/28)
 - Avoid double-building CI on PRs. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/29)
 - Fix Travis coveralls post-build step. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/144)
 - Add ShellCheck to Travis build. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/150)
 - [need to redo] Add GitHub Actions CI builds. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/182)
 - Add GitHub Actions CI builds. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/183)
 - Installation of the SpinalCord Toolbox on GH Actions. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/184)
 - Remove extraneous workflow. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/202)
 - Cache SCT in GitHub Actions CI workflow. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/225)

**INSTALLATION**
 - ci: Add pre-commit configs. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/215)
 - CI: Check for correct EOF. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/223)
 - ci: Check for end-of-line trailing whitespace. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/224)

**BUG**
 - Fix coveralls 3.0.0 bug and github actions macos 11.0 bug. [View pull request](https://github.com/shimming-toolbox/shimming-toolbox/pull/207)
