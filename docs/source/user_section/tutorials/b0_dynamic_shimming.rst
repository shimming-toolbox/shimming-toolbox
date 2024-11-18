.. _dynamic_shimming:

Dynamic Shimming Tutorial
-------------------------

In this tutorial, we will learn the following:

- Launch ``Shimming-Toolbox``'s GUI
- Create a fieldmap
- Create a mask
- Perform dynamic shimming

Download test data
~~~~~~~~~~~~~~~~~~

Open a Terminal and download this small dataset by running the following command:

.. code:: bash

    st_download_data data_dynamic_shimming

Go to the dataset folder:

.. code:: bash

  cd data_dynamic_shimming


Launch the plugin
~~~~~~~~~~~~~~~~~

In your terminal, run the command:

.. code:: bash

    shimming-toolbox

The plugin should open as a new panel in ``FSLeyes``.


Create a Fieldmap
~~~~~~~~~~~~~~~~~

- Load the phase echo(es) and the first magnitude NIfTI files in FSLeyes.

  - data_dynamic_shimming/sub-spine/fmap/sub-spine_magnitude1.nii.gz
  - data_dynamic_shimming/sub-spine/fmap/sub-spine_phase2.nii.gz

- Navigate to the *Fieldmap* Tab. If you don't see the tab, drag the right edge of the ``Shimming Toolbox`` panel to make all the tabs appear.
- Enter 1 for the *Number of Echoes*.
- Select the phase from the overlay and click the *Input Phase 1* button.
- Select the first magnitude image in the overlay and click the *Input Magnitude*.
- *(Optional)* Change the output file and folder by clicking on *Output File*.
- Click *Run*.
- The output fieldmap should load automatically.

Create a Mask
~~~~~~~~~~~~~

- Load the target anatomical image.

  - data_dynamic_shimming/sub-spine/anat/sub-spine_unshimmed_e1.nii.gz

- Select The *Mask* Tab.
- Select *Box* from the dropdown.
- Select the target image from the overlay, click the button *Input*.
- Input voxel indexes for *center* and *size*. TIP: Look at the Location panel of fsleyes to locate the center of the ROI.
  - For the spine, a *center* of 140, 124, 6 and a *size* of 30, 15, 12 could work.
- *(Optional)* Change the output file and folder by clicking on *Output File*.
- Click *Run*.
- The output mask should load automatically.

Dynamic shimming
~~~~~~~~~~~~~~~~

- Navigate to the *B0 Shim* Tab.
- Select *Dynamic* in the dropdown menu (it should already be selected by default).
- Select the fieldmap from the overlay, click the button *Input Fieldmap*.
- Select the target anatomical image, click the button *Input Anat*.
- Select the mask, click the button *Input Mask*.
- Select a *Slice Ordering* of Ascending.
- Select a *Slice Factor* of 1 (should be the default).
- Select a *Scanner Order* of 1. It means that dynamic shimming will be
  performed with the linear gradients of the scanner. In typical scanners, order 2
  or higher is not compatible with dynamic shimming, due to the high inductance of the
  shim coils (they cannot be updated as rapidly as the gradient coils).
- *(Optional)* Change the output folder by clicking the *Output Folder* button.
- Click *Run*.
- The output text files and figures should be in the *Output Folder*. You can
  then copy the text files onto the MRI console to be read by the pulse sequence.
