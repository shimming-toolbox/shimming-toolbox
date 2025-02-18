.. _b1_shimming:

Static B1+ shimming
-------------------
.. warning::
    This tutorial only covers static B1+ shimming applications performed on Siemens scanners. Shimming-Toolbox currently
    only handles B1+ maps acquired using Siemens' standard TurboFLASH B1+ mapping sequence.
    This is the tfl_rfmap sequence, which can be found under Siemens/Service Sequences in the Protocols. Not to be confused
    with the tfl_b1map sequence, which will not create the channel-specific B1+ maps necessary for the shim process.

Download test data
~~~~~~~~~~~~~~~~~~

From your terminal, "cd" to any folder and run:

.. code:: bash

    st_download_data data_b1_shimming

This will download the example B1+ maps dataset in the folder from which you typed this command.

Start the GUI of Shimming Toolbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your terminal, run:

.. code:: bash

    shimming-toolbox

The plugin should open as a new panel in ``FSLeyes``.

Convert DICOM to NIfTI
~~~~~~~~~~~~~~~~~~~~~~

In FSLeyes, click on the ``dicom_to_nifti`` tab.

.. note::
    If you don't see the tab, drag the right edge of the ``Shimming Toolbox`` panel to make all the tabs appear.

- Click on *Input Folder* to select the downloaded ``data_b1_shimming/dicoms`` path as an input.
- Enter a subject name (e.g. "test").
- Modify the output folder path a new folder called "niftis" in the previously downloaded "data_b1_shimming" folder: ``<your-local-path>/data_b1_shimming/niftis``.
- Click *Run*.
- A B1+ map should automatically load in the overlay.

Create a Mask
~~~~~~~~~~~~~

In an actual experiment, a mask would probably be created from an anatomical image using a segmentation tool.
However, in this tutorial, we will create a simple box mask from the B1+ acquisition.
Since the B1+ acquisition has complex 4D B1+ data, we first convert it to a magnitude image and compute the average
over the last dimension so that it can be used by the masking pipeline.
We are then ready to create a box mask from that 3D image.

In your terminal where you downloaded the b1 dataset, run:

.. code:: bash

    st_maths mag --complex data_b1_shimming/niftis/sub-test/rfmap/sub-test_TB1map_uncombined.nii.gz --output data_b1_shimming/niftis/derivatives/sub-test_TB1map_uncombined_mag.nii.gz
    st_maths mean --input data_b1_shimming/niftis/derivatives/sub-test_TB1map_uncombined_mag.nii.gz --output data_b1_shimming/niftis/derivatives/sub-test_TB1map_uncombined_mean.nii.gz

- Load the ``data_b1_shimming/niftis/derivatives/sub-test_TB1map_uncombined_mean.nii.gz`` image in FSLeyes (drag and drop).
- Select the *Mask* Tab.
- Select *Box* from the dropdown.
- Select the ``sub-test_TB1map_uncombined_mean`` image from the overlay, click the button *Input*.
- Input voxel indexes for *center* (suggestion: 48, 48, 19) and *size* (suggestion: 35, 50, 12). Look at the Location panel of FSLeyes to locate the center of the ROI.
- *(Optional)* Change the output file and folder by clicking on *Output File*.
- Click *Run*.
- The output mask should load automatically in the *Overlay list*.

Static B1+ shimming: CV reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*(Optional)* The SarDataUser.mat is a hard requirement for the "SAR efficiency" shim option, and an optional requirement for CV reduction. To ensure that the SarDataUser.mat on the scanner contains the VOPs of the coil currently being used, create a copy of the anatomical scan used to create the mask, and start the scan. The scan can be stopped once data acquisition starts. In the Exam Card, under System/pTx Volumes, set the B1 shim mode to "Patient Specific". See the figure below for the location of the switch

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/B1shim_button.jpg
  :width: 400
  :alt: B1 shim option location

Copy the SarDataUser.mat file from C:/Medcom/MriProduct/PhysConfig/ to the laptop on which Shimming Tooolbox is run

- Navigate to the *B1+ Shim* Tab.
- Select *CV reduction* in the dropdown menu (it should already be selected by default).
- Select the uncombined B1+ maps from the overlay, click the button *Input B1+ map*.
- Select the mask, click the button *Input Mask*.
- *(Optional)* If you have a SarDataUser.mat VOP file, you can locate it after clicking on *input VOP file*. You can then adjust the SAR factor to indicate by how much your
  optimized shim weights might exceed the max local SAR of a phase only shimming.
- Click *Run*.
- The output text file containing the shim weights should be in the *Output Folder*.
- Manually input these shim weights on the scanner console. On Siemens scanners, input them in the
  ``Options > Adjustments > B1 Shim`` window. Make sure to also set ``System > pTx Volumes > B1 Shim mode`` to
  *"Patient-specific"* in the sequence parameters to ensure that the
  shim-weights will be applied during the acquisition.
