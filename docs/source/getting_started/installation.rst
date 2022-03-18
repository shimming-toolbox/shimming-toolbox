.. _installation:

************
Installation
************

``Shimming Toolbox`` is written in Python. It can be used either from the Terminal
or from a graphical user interface (GUI) as a plugin for `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`__.

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/fsleyes_example.png
  :alt: Overview
  :width: 1000


Dependencies
------------

``Shimming Toolbox`` works on ``macOs`` and ``Linux`` operating systems. There is a plan to support
``Windows`` in the future.

**Optional dependencies:**

- If you would like to use ``prelude`` for phase unwrapping and/or ``bet`` for brain extraction, you need to install `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__.
- If you would like to use ``sct_deepseg_sc`` for spinal cord segmentation, you need to install `SCT <https://spinalcordtoolbox.com/>`__.


Installation
------------

The installer will automatically install:

- ``FSLeyes``
- ``Shimming Toolbox``
- ``dcm2niix``

Open a Terminal and run the following commands.

First, download the FSLeyes plugin:

.. code:: bash

    git clone https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox.git

Next, run the installer:

.. code:: bash

    cd fsleyes-plugin-shimming-toolbox
    make install

You will be prompted to source your ``*.shrc`` file. For example:

.. code:: bash

    source ~/.bashrc

Now, you can launch ``FSLeyes`` with our plugin from any environment:

.. code:: bash

    shimming-toolbox

To launch our plugin, go to:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/open_st_fsleyes.png
  :alt: Overview
  :width: 1000

The plugin should open as a panel.


Test the Installation
---------------------

This step is optional but it's a good measure to ensure
``Shimming Toolbox`` is properly installed on your system.

.. warning::
  The testing section of the documentation is still work in progress.


Comprehensive Test
~~~~~~~~~~~~~~~~~~

To run the entire testing suite, run ``pytest`` from the
**cloned** shimming-toolbox directory:

.. code:: bash

  cd ~/shimming-toolbox/shimming-toolbox
  source $HOME/shimming-toolbox/python/etc/profile.d/conda.sh
  conda activate st_venv
  pytest

See https://docs.pytest.org/ for more options.

If all tests pass, ``Shimming Toolbox`` is properly installed.


Testing subsets of soft dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prelude`` and ``dcm2niix`` are soft dependencies, so you may wish to run the
parts of the testing suite that do not depend on them.

To test shimming-toolbox without ``prelude`` and without ``dcm2niix``:

.. code:: bash

  cd ~/shimming-toolbox/shimming-toolbox
  source $HOME/shimming-toolbox/python/etc/profile.d/conda.sh
  conda activate st_venv
  pytest -m "not prelude and not dcm2niix"

To test shimming-toolbox without ``prelude`` and with ``dcm2niix``, you can use the above block but modifying the ``-m`` argument to ``"not prelude"``.

To test shimming-toolbox with ``prelude`` and without ``dcm2niix``, you can use the above block but modifying the ``-m`` argument to ``"not dcm2niix"``.

To test **only** the parts of shimming-toolbox dependent on ``prelude`` or
``dcm2niix``, the corresponding ``-m`` argument is ``"prelude or dcm2niix"``

Note that supplying the ``"-m"`` argument ``"prelude and dcm2niix"`` only runs tests dependent on both ``prelude`` **and** ``dcm2niix``.


For Developers
--------------

The installation files can be found in the ``installer`` folder, and are called by the ``Makefile``.

When you run ``make install``, we first check if the ``ST_DIR`` exists, or if a clean install has
been requested. The ``ST_DIR`` is where this package and also the ``fsleyes-plugin-shimming-toolbox`` are installed. By choosing clean, you delete the entire install directory, and consequently any prior installs of ``shimming-toolbox`` or ``fsleyes-plugin-shimming-toolbox``. Note that this is set to ``CLEAN==false`` by default.

We next check if ``conda`` has been installed into the ``ST_DIR``. If not, we run the ``conda`` installer.

Finally, we create a virtual environment and install ``shimming-toolbox``.
