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
- Select the phase image in the overlay (**sub-spine_phase2**) and click the *Input Phase 1* button.
- Select the first magnitude image in the overlay (**sub-spine_magnitude1**) and click the *Input Magnitude*.
- Select the unwrapper you wish to use. Select skimage if you do not have prelude installed (External dependency: FSL).
- *(Optional)* Change the output file and folder by clicking on *Output File*.
- Click *Run*.
- The output fieldmap should load automatically.

.. admonition:: Note

  The command line interface (CLI) equivalent of the steps performed above can be done using `st_prepare_fieldmap <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#st-prepare-fieldmap>`__.
  See that page for the list of the different arguments/options available and how to use them. For convenience, the CLI options corresponding to the some of the steps above are:
  `--mag <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_prepare_fieldmap-mag>`__,
  `--unwrapper <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_prepare_fieldmap-unwrapper>`__,
  `--output <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_prepare_fieldmap-o>`__.

Create a Mask
~~~~~~~~~~~~~

- Load the target image.

  - data_dynamic_shimming/sub-spine/anat/sub-spine_unshimmed_e1.nii.gz

- Select The *Mask* Tab.
- Select *Box* from the dropdown.
- Select the target image in the overlay (**sub-spine_unshimmed_e1**), then click the button *Input*.
- Input voxel indexes for *center* and *size*. TIP: Look at the Location panel of fsleyes to locate the center of the ROI.

  - For the spine, a *center* of 128, 124, 6 and a *size* of 30, 15, 12 could work.
  - TIP: You can use external tools such as `Spinal Cord Toolbox <https://github.com/spinalcordtoolbox/spinalcordtoolbox>`__ (SCT) to create spinal cord masks automatically.

- *(Optional)* Change the output file and folder by clicking on *Output File*.
- Click *Run*.
- The output mask should load automatically.

.. admonition:: Note

  The CLI equivalent of the steps performed above can be done using `st_mask box <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#st-mask-box>`__.
  For convenience, the CLI options corresponding to some of the steps above are:
  `--input <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_mask-box-i>`__,
  `--center <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_mask-box-center>`__,
  `--size <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_mask-box-size>`__,
  `--output <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_mask-box-o>`__.

Dynamic shimming
~~~~~~~~~~~~~~~~

- Navigate to the *B0 Shim* Tab.
- Select *Dynamic/volume* in the dropdown menu (it should already be selected by default).
- Select the fieldmap in the overlay and click the button *Input Fieldmap*.
- Select the target image in the overlay (**sub-spine_unshimmed_e1**), then click the button *Input target*.
  The target image is used to know the slice geometry when using slice-wise shimming.
- Select the mask in the overlay and click the button *Input Mask*.
- Select a *Slice Ordering* of Ascending. This option will perform slice-wise shimming and select an acquisition slice order of ascending. Selecting Volume with perform a volume shim. Selecting all other options will perform a slice-wise shim.
- Select a *Slice Factor* of 1 (should be the default). Relevant when using multi band acquisitions.
- Select the 0 and 1 *Scanner Order* checkboxes. Shimming will be
  performed with the frequency and the linear gradients of the scanner. In typical scanners, order 2
  or higher is not compatible with dynamic shimming, due to the high inductance of the
  shim coils (they cannot be updated as rapidly as the gradient coils).
- *(Optional)* Change the output folder by clicking the *Output Folder* button.
- Click *Run*.
- The output text files and figures should be in the *Output Folder*. You can
  then copy the text files onto the MRI console to be read by the pulse sequence.

.. admonition:: Note

  The CLI equivalent of the steps performed above can be done using `st_b0shim dynamic <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#st-b0shim-dynamic>`__.
  For convenience, the CLI options corresponding to some of the steps above are:
  `--fmap <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-fmap>`__,
  `--target <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-target>`__,
  `--mask <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-mask>`__,
  `--slices <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-slices>`__,
  `--slice-factor <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-slice-factor>`__,
  `--scanner-coil-order <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-scanner-coil-order>`__,
  `--output <https://shimming-toolbox.org/en/latest/cli_reference/cli.html#cmdoption-st_b0shim-dynamic-o>`__.
