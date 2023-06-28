.. _b1_shimming:

Static B1+ shimming
-------------------
.. warning::
    This tutorial only covers static B1+ shimming applications performed on Siemens scanners. Shimming-Toolbox currently
    only handles B1+ maps acquired using Siemens' standard TurboFLASH B1+ mapping sequence.
    This is the tfl_rfmap sequence, which can be found under Siemens/Service Sequences in the Protocols. Not to be confused      
    with the tfl_b1map sequence, which will not create the channel-specific B1+ maps necessary for the shim process.

Open a Terminal and run:

.. code:: bash

    shimming-toolbox

Then, open the ``Shimming Toolbox`` plugin:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

The plugin should open as a new panel in ``FSLeyes``.

Download test data
~~~~~~~~~~~~~~~~~~

From your terminal, cd to any folder and run:

.. code:: bash

    st_download_data data_b1_shimming

This will download the example B1+ maps dataset in the folder from which you typed this command.

Convert DICOM to NIfTI
~~~~~~~~~~~~~~~~~~~~~~

In FSLeyes, click on the ``dicom_to_nifti`` tab.

.. note::
    If you don't see the tab, drag the right edge of the ``Shimming Toolbox`` panel to make all the tabs appear.


- Click on *Input Folder* to select the downloaded ``data_b1_shimming/dicoms`` path as an input.
- Enter a subject name (e.g. "test").
- *(optional)* Modify the output folder.
- Click *Run*.

Create a Mask
~~~~~~~~~~~~~

- Load the target anatomical image.
- Select The *Mask* Tab.
- Select *Box* from the dropdown.
- Select the target image from the overlay, click the button *Input*.
- Input voxel indexes for *center* and *size*. Look at the Location panel of FSLeyes to locate the center of the ROI.
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
