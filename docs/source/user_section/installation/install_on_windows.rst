***********************
Installation on Windows
***********************

``Shimming Toolbox`` can be used from the Terminal on Windows system.

.. code-block:: console

  $ st_prepare_fieldmap -h

  Usage: st_prepare_fieldmap [OPTIONS] PHASE...

  Creates fieldmap (in Hz) from phase images.

  This function accommodates multiple echoes (2 or more) and phase difference.
  This function also accommodates 4D phase inputs, where the 4th dimension
  represents the time, in case multiple field maps are acquired across time
  for the purpose of real-time shimming experiments. For non Siemens phase
  data, see --autoscale-phase option.

  PHASE: Input path of phase nifti file(s), in ascending order: echo1, echo2,
  etc.

  Options:
  --mag PATH                     Input path of mag nifti file  [required]
  --unwrapper [prelude|skimage]  Algorithm for unwrapping. skimage is
                                 installed by default, prelude requires FSL to
                                 be installed.  [default: prelude]
  -o, --output PATH              Output filename for the fieldmap, supported
                                 types : '.nii', '.nii.gz'  [default:
                                 ./fieldmap.nii.gz]
  --autoscale-phase BOOLEAN      Tells whether to auto rescale phase inputs
                                 according to manufacturer standards. If you
                                 have non standard data, it would be
                                 preferable to set this option to False and
                                 input your phase data from -pi to pi to avoid
                                 unwanted rescaling  [default: True]
  --mask PATH                    Input path for a mask. Mask must be the same
                                 shape as the array of each PHASE input.
  --threshold FLOAT              Threshold for masking if no mask is provided.
                                 Allowed range: [0, 1] where all scaled values
                                 lower than the threshold are set to 0.
                                 [default: 0.05]
  --savemask PATH                Filename of the mask calculated by the
                                 unwrapper
  --gaussian-filter BOOLEAN      Gaussian filter for B0 map
  --sigma FLOAT                  Standard deviation of gaussian filter. Used
                                 for: gaussian_filter
  -v, --verbose [info|debug]     Be more verbose
  -h, --help                     Show this message and exit.


Dependencies
------------

**Optional dependencies:**

- ``prelude`` (FSL) is only supported on macOS and Linux. To use prelude, install Shimming Toolbox on a macOS or Linux system.
- If you would like to use ``sct_deepseg_sc`` for spinal cord segmentation, you need to install `SCT <https://spinalcordtoolbox.com/>`__.


Installation Procedure for Windows
----------------------------------

Conda requires an installation path that does not contain spaces. Your username should not contain spaces to avoid this issue.
If your username contains spaces, you can create a new user account with no spaces in the username or use a different OS.

.. Note::

    The installer will install ``Shimming Toolbox`` and ``dcm2niix`` into an isolated environment. It will not interfere if you already have ``dcm2niix`` installed.

Open a ``cmd`` command prompt. Navigate to where you want to download Shimming Toolbox and run the following commands.

First, download Shimming Toolbox:

.. code:: bat

    git clone https://github.com/shimming-toolbox/shimming-toolbox.git

Next, run the installer:

.. code:: bat

    cd shimming-toolbox
    installer\windows_install.bat

.. Warning::
    To use Shimming Toolbox's scripts, you will be prompted to either reboot your computer or follow these instructions:

    Open the Start Menu -> Type 'environment' -> Open **Edit environment variables for your account**

    Click 'OK'

To make sure the installation was successful, run the following command:

.. code:: bat

    st_b0shim --help

Test the Installation
---------------------

This step is optional but it's a good measure to ensure
``Shimming Toolbox`` is properly installed on your system.


Comprehensive Test
~~~~~~~~~~~~~~~~~~

To run the testing suite, run ``pytest`` from the shimming-toolbox source directory:

.. code:: bat

  cd <shimming-toolbox-dir>/shimming-toolbox
  %userprofile%\shimming-toolbox\python\Scripts\activate
  pytest -m "not prelude"

See https://docs.pytest.org/ for more options.

If all tests pass, ``Shimming Toolbox`` is properly installed with all supported dependencies (SCT).

Testing subsets of soft dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SCT`` is a soft dependencies, so you may wish to run the
parts of the testing suite that do not depend on it.

To test shimming-toolbox without ``SCT``:

.. code:: bat

  cd <shimming-toolbox-dir>/shimming-toolbox
  %userprofile%\shimming-toolbox\python\Scripts\activate
  pytest -m "not prelude and not sct"

To test **only** the parts of shimming-toolbox dependent on ``sct``, the corresponding ``-m`` argument is ``"sct"``

For Developers
--------------

The installation script can be found in the ``installer`` folder as ``windows_installer.bat``.

When you run the installer, we first check if the ``ST_DIR`` exists. The ``ST_DIR`` is where the ``shimming-toolbox`` package.

We then install ``conda``. Next, we install ``shimming-toolbox`` into the base environment of the new conda installation.
