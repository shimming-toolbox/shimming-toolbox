.. _b1_shimming:

Static B1+ shimming
-------------------

Open a Terminal and run:

.. code:: bash

    shimming-toolbox

Then, open the ``Shimming Toolbox`` plugin:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

The plugin should open as a new panel in ``FSLeyes``.

Download test data
~~~~~~~~~~~~~~~~~~

From your terminal, cd to any folder and type.

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

Static B1+ shimming
~~~~~~~~~~~~~~~~~~~

- Navigate to the *B1+ Shim* Tab.
- Select *CV reduction* in the dropdown menu (it should already be selected by default).
- Select the uncombined B1+ maps from the overlay, click the button *Input B1+ map*.
- Select the mask, click the button *Input Mask*.
- *(Optional)* If you have a SarDataUser.mat VOP file (found on the scanner in C:/Medcom/MriProduct/PhysConfig.), you
  can locate it after clicking on *input VOP file*. You can then adjust the SAR factor to indicate by how much your
  optimized shim weights might exceed the max local SAR of a phase only shimming.
- Click *Run*.
- The output text file containing the shim weights should be in the *Output Folder*. You then need to enter these shim
  weights on the scanner console in the ``Options > Adjustments > B1 Shim`` window. Make sure to also select
  *"Patient-specific"* in the ``System > pTx Volumes > B1 Shim mode`` in the sequence parameters to ensure that the
  shim-weights will be applied.
