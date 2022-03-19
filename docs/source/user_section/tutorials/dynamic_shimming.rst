.. _dynamic_shimming:

Dynamic Shimming
----------------

Open a Terminal and run:

.. code:: bash

    shimming-toolbox

Then, open ``Shimming Toolbox`` plugin:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

The plugin should open as a new panel in ``FSLeyes``.

Download test data
~~~~~~~~~~~~~~~~~~

.. warning::
  TODO

Go to the dataset folder:

.. warning::
  TODO


Convert DICOM to NIfTI
~~~~~~~~~~~~~~~~~~~~~~

Click on the tab ``dicom_to_nifti``

.. note::
  If you don't see the tab, drag the right edge of the ``Shimming Toolbox`` panel
  to make all the tabs appear.


.. warning::
  TODO


Create a Fieldmap
~~~~~~~~~~~~~~~~~

- Load the phase echo(es) and the first magnitude NIfTI files.
- Navigate to the *Fieldmap* Tab
- Enter 1 for the *Number of Echoes*
- Select the phase from the overlay and click the *Input Phase 1* button
- Select the first magnitude image in the overlay and click the *Input Magnitude*
- *(Optional)* Change the output file and folder by clicking on *Output File*
- Click *Run*
- The output fieldmap should load automatically

Create a Mask
~~~~~~~~~~~~~

- Load the target anatomical image
- Select The *Mask* Tab
- Select *Box* from the dropdown
- Select the target image from the overlay, click the button *Input*
- Input voxel indexes for *center* and *size*. TIP: Look at the Location panel of fsleyes to locate the center of the ROI
- *(Optional)* Change the output file and folder by clicking on *Output File*
- Click *Run*
- The output mask should load automatically

Dynamic shimming
~~~~~~~~~~~~~~~~

- Navigate to the *B0 Shim* Tab
- Select *Dynamic* in the dropdown menu (it should already be selected by default).
- Select the fieldmap from the overlay, click the button *Input Fieldmap*
- Select the anatomical image, click the button *Input Anat*
- Select the mask, click the button *Input Mask*
- Select a *Slice Ordering* of Sequential (should be the default).
- Select a *Slice Factor* of 1 (should be the default).
- Select a *Scanner Order* of 1. It means that dynamic shimming will be
  performed with the linear gradients of the scanner. In typical scanners, order 2
  or higher is not compatible with dynamic shimming, due to the high inductance of the
  shim coils (they cannot be updated as rapidly as the gradient coils).
- *(Optional)* Change the output folder by clicking the *Output Folder* button
- Click *Run*
- The output text files and figures should be in the *Output Folder*. You can
  then copy the text files onto the MRI console to be read by the pulse sequence.
