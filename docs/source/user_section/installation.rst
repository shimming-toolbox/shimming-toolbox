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

.. Note::

    The installer will install ``FSLeyes`` and ``dcm2niix`` into isolated environments. They will be used by
    ``Shimming Toolbox`` and will not interfere if you already have ``FSLeyes`` or ``dcm2niix`` installed. If you
    have previously installed ``FSLeyes``, it will not support the GUI.

Open a Terminal and run the following commands.

First, download the FSLeyes plugin:

.. code:: bash

    git clone https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox.git

Next, run the installer:

.. code:: bash

    cd fsleyes-plugin-shimming-toolbox
    make install

You will be prompted to source your ``.*shrc`` file. For example:

.. code:: bash

    source ~/.bashrc

.. Note::

    You can restart your terminal or open a new tab to source your ``.*shrc`` file automatically.


The ``shimming-toolbox`` command launches FSLeyes with GUI support.

.. code:: bash

    shimming-toolbox

To launch our plugin, go to:

.. code:: bash

    Settings --> OrthoView --> Shimming Toolbox

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/open_st_fsleyes.png
  :alt: Overview
  :width: 1000

The plugin should open as a panel.

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/st_fsleyes_plugin.png
  :alt: Overview
  :width: 1000

Test the Installation
---------------------

This step is optional but it's a good measure to ensure
``Shimming Toolbox`` is properly installed on your system.


Comprehensive Test
~~~~~~~~~~~~~~~~~~

To run the entire testing suite, run ``pytest`` from the
**cloned** shimming-toolbox directory:

.. code:: bash

  cd ~/shimming-toolbox/shimming-toolbox
  source $HOME/shimming-toolbox/python/etc/profile.d/conda.sh
  conda activate ~/shimming-toolbox/python/     
  pytest

See https://docs.pytest.org/ for more options.

If all tests pass, ``Shimming Toolbox`` is properly installed.


Testing subsets of soft dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prelude`` is a soft dependencies, so you may wish to run the
parts of the testing suite that do not depend on it.

To test shimming-toolbox without ``prelude``:

.. code:: bash

  cd ~/shimming-toolbox/shimming-toolbox
  source $HOME/shimming-toolbox/python/etc/profile.d/conda.sh
  conda activate ~/shimming-toolbox/python/
  pytest -m "not prelude"

To test **only** the parts of shimming-toolbox dependent on ``prelude``, the corresponding ``-m`` argument is ``"prelude"``

For Developers
--------------

The installation files can be found in the ``installer`` folder, and are called by the ``Makefile``.

When you run ``make install``, we first check if the ``ST_DIR`` exists, or if a clean install has
been requested. The ``ST_DIR`` is where the ``shimming-toolbox`` package and also the ``fsleyes-plugin-shimming-toolbox`` are installed. By choosing clean, you delete the entire install directory, and consequently any prior installs of ``shimming-toolbox`` or ``fsleyes-plugin-shimming-toolbox``. Note that this is set to ``CLEAN==false`` by default.

We next check if ``conda`` has been installed into the ``ST_DIR``. If not, we run the ``conda`` installer.

Next, we create a virtual environment for ``fsleyes-plugin-shimming-toolbox`` and install the package into it.

Finally, we create a virtual environment for ``shimming-toolbox`` and install the package into it.

.. Note::

    You can track the Github version of ``shimming-toolbox`` if you are a developer. This will remove any previous install of ``shimming-toolbox``,
    and replace it with the version you have cloned. *Note that this may break the plugin* since you are using a version
    that has not been tested on the plugin. You can install ``shimming-toolbox`` development version with the following steps:

.. code:: bash

    git clone https://github.com/shimming-toolbox/shimming-toolbox.git
    cd shimming-toolbox
    make install

You will be prompted to source your ``.*shrc`` file. For example:

.. code:: bash

    source ~/.bashrc

You can then activate the ``shimming-toolbox`` environment and start coding!

.. code:: bash

    conda activate ~/shimming-toolbox/python/
