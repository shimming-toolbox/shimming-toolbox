.. include:: ../../../examples/README.rst

GUI Dynamic Shimming Scenario
-----------------------------

After following the :ref:`installation procedure of the plugin<installation_gui>`.

You can launch ``FSLeyes`` with our plugin from any environment:

.. code:: bash

    shimming-toolbox

To launch our plugin, go to:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

The plugin should open as a panel.

Create a Fieldmap
~~~~~~~~~~~~~~~~~

- Load the phase echo(es) and the first magnitude NIfTI files in ``FSLeyes``
- Navigate to the *Fieldmap* Tab
- Enter 1 for the *Number of Echoes*
- Select the phase from the overlay and click the *Input Phase 1* button
- Select the magnitude in the overlay and click the *Input Magnitude*
- (Optional) Change the output file and folder by clicking the *Output File* button
- Select *Run*
- The output fieldmap should load in automatically

Create a Mask
~~~~~~~~~~~~~

- Load the target anatomical image
- Select The *Mask* Tab
- Select *Box* from the dropdown
- Select the target image from the overlay, click the button *Input*
- Input voxel indexes for *center* and *size*. TIP: Look at the Location panel of fsleyes to locate the center of the ROI
- (Optional) Change the output file and folder by clicking the *Output File* button
- Select *Run*
- The output mask should load in automatically

Dynamic shim
~~~~~~~~~~~~

- Load the fieldmap, target image and the mask NIfTI files in ``FSLeyes``
- Navigate to the *B0 shim* Tab
- Select the fieldmap from the overlay, click the button *Input Fieldmap*
- Repeat for the target anatomical image and the mask
- Select a *Scanner Order* of 1
- Select a *Slice Ordering* of sequential and a *Slice Factor* of 1
- (Optional) Change the output folder by clicking the *Output Folder* button
- Select *Run*
- The output text files and figures should be in the *Output Folder*
