# Client facing messages
# English 


# Messages, error or otherwise
# Mask commandline messages are at the bottom of the file

_array_3d = 'Input arrays X, Y, and Z must be 3d'
_array_8_shape = 'Input arrays should have 4th dimension\'s shape equal to 8'
_copy_dicom_failure = 'Copy of original dicom files has failed'
_dcm2bids_helper_creation ='dcm2bids_helper could not create directory helper'
_dimensions_input = 'Dimensions of input can only be 2D, 3D or 4D'
_dir_files_input = 'Directories and files in input path'
_download_error = 'Download error'
_echo_point_numbers = 'Phasediff must have 2 echotime points. Otherwise the number of echoes must match the'
_failed_dcm2bids_helper = 'Could not run dcm2bids_helper'
_identical_sizing = 'Input arrays X, Y, and Z must be identically sized'
_input_phase_difference = "The JSON file of the input phase should include the fields EchoTime1 and EchoTime2 if it is a phase difference."
_image_3d = 'Anatomical image must be in 3d'
_incorrect_fmap = 'fmap must be 4d x, y, z, t'
_input_date_format = 'Input format does not follow HHMMSS.mmmmmm'
_json_formatting = 'Errors in JSON file syntax'
_json_missing = 'Missing json file'
_json_missing_fields = 'JSON file missing required key/value pairs'
_mag_data_parsing = 'Mag data parsing is wrongly parsed'
_mag_phase_dimension = 'mag and phase must have the same dimensions.'
_mask_anat_match = 'Mask must have the same shape and affine transformation as anat'
_mask_phase = 'Shape of mask and phase must match.'
_nifty_2d_3d = 'The nifti file does not have 2 or 3 dimensions.'
_nifty_3d = 'The nifti file does not have 3 dimensions.'
_no_bids_structure = 'Creating bids structure for data failed'
_no_data = 'No data to process'
_no_existing_nifty_path = 'Not an existing NIFTI path'
_no_dcm2bids = 'Cannot call dcm2bids'
_no_dicom_config = 'No dcm2bids config file found at ' 
_no_dicom_path = 'No dicom path found at '
_no_nifty_file = 'Could not create a nifti file'
_phase_data_parsing = 'Phase data parsing is wrongly parsed'
_phase_number = 'This number of phase input is not supported: '
_pi_range = 'read_nii must range from -pi to pi.'
_pos_order = 'Orders must be positive'
_quiet = ''
_resp_trace_time_limit = 'acquisition_times do not fit within time limits for resp trace'
_same_magnitude_wrapped_phase = 'The magnitude image mag must be the same shape as wrapped_phase'
_same_mask_wrapped_phase = 'Mask must be the same shape as wrapped_phase'
_square_2d = 'shape_square only allows for 2 dimensions'
_temp_removal = 'Could not remove tmp file'
_unimplemented_unwrap = 'This unwrap function is not implemented:'
_unsupported_phase = 'Shape of input phase is not supported.'
_url_filename = 'Unable to determine target filename for URL: '
_wrapped_2d_3d = 'Wrapped_phase must be 2d or 3d'

# Mask Commandline Messages
_mask_box_2d = "(int): Length of the side of the box along first and second dimension (in pixels). (nargs=2)"
_mask_box_3d = "(int): Length of the side of the box along first, second and third dimension (in pixels). \n(nargs=3)"
_mask_box = "(int): Center of the box along first and second dimension (in pixels). If no center is \nprovided (None), the middle is used. (nargs=2) (default: None, None)"
_mask_centre = "(int): Center of the box along first, second and third dimension (in pixels). If no center\nis provided (None), the middle is used. (nargs=3) (default: None, None, None)"
_mask_group_help = """Create a mask based on a specified shape (box, rectangle, SpinalCord Toolbox mask) or based on the 
                   thresholding of an input image. Callable with the prefix 'st' in front of 'mask'. 
                   (Example: 'st_mask -h')."""
_mask_help =       """Create a box mask from the input file. The nifti file is converted to a numpy array. If this
                   array is in 3D dimensions, then a binary mask is created from this array in the form of a bo
                   with lengths defined in 'size'. This box is centered according to the 3 dimensions indicated
                   in 'center'. The mask is stored by default under the name 'mask.nii.gz' in the output folder.
                   Return the filename for the output mask."""
_mask_input_3D =      "(str): Input path of the nifti file to mask. This nifti file must have 3D. Supported extensions are \n.nii or .nii.gz."
_mask_path_input_2D3D =          "(str): Input path of the nifti file to mask. This nifti file must have 2D or 3D. Supported\nextensions are .nii or .nii.gz."
_mask_input_thresh = "(str): Input path of the nifti file to mask. Supported extensions are .nii or .nii.gz."
_mask_output = "(str): Name of output mask. Supported extensions are .nii or .nii.gz. (default: \n(os.curdir, 'mask.nii.gz'))"
_mask_output_filemask =     "The filename for the output mask is: "
_mask_rectange_from_input = """Create a rectangle mask from the input file.
                            The nifti file is converted to a numpy array. If this array is in 2 dimensions, then a binary
                            mask is created from this array in the form of a rectangle of lengths defined in 'size'. This
                            rectangle is centered according to the 2 dimensions indicated in 'center'. If this array is
                            in 3 dimensions, a binary mask is created in the shape of rectangle for each slice of the 3rd
                            dimension of the array, in the same way as for a 2D array. The masks of all these slices are
                            grouped in an array to form a binary mask in 3 dimensions. The mask is stored by default under
                            the name 'mask.nii.gz' in the output folder."
                            Return an output nifti file with square mask."""
_mask_threshold = """Create a threshold mask from the input file. "
                  The nifti file is converted into a numpy array. A binary mask is created from the thresholding
                  of the array. The mask is stored by default under the name 'mask.nii.gz' in the output 
                  folder. Return an output nifti file with threshold mask."""
_mask_threshold_value = "(int): Value to threshold the data: voxels will be set to zero if their \nvalue is equal or less than this threshold. (default: 30)"
                 
                 
                 
                 
                   









                            
                            

                   
                   
                   
                   

              





                       





                                       


















                   

