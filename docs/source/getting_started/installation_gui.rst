.. _installation_gui:

*************************
Installing FSLeyes Plugin
*************************

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/fsleyes_example.png
  :alt: Overview
  :width: 1000

`FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`__ is a neuroimaging viewer, created
as part of the `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`__ software package. We have created a `plugin <https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox>`__ for ``Shimming Toolbox`` to
integrate with ``FSLeyes``, which means that our software can be loaded and used with their
viewer.

We have bundled our plugin installer with ``FSLeyes``, which means you don't have to install it
yourself. If you do have ``FSLeyes`` installed, don't worry - it won't interfere.

To install, first clone the repo:

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

The plugin should open as a panel.
