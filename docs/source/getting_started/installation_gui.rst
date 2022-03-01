.. _installation_gui:

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

..
  TODO: ADD IMAGE

The plugin should open as a panel.
