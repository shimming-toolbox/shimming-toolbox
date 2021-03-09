**************
FSLeyes Plugin
**************

Installation
============

First, you will need to make sure ``shimming-toolbox`` is installed using a ``conda``
environment, as detailed here in the ``Installation`` section.

Next, make sure ``dcm2niix`` is installed to version `v1.0.20201102 <https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20201102>`_ or later.
You could install it via conda:

.. code-block::

   yes | conda install -c conda-forge/label/main dcm2niix

Then navigate to the Shimming Toolbox folder and activate your Shimming Toolbox environment:

.. code-block::

   cd shimming-toolbox
   conda activate shim_venv

Next, install ``wxPython`` using ``conda-forge``:

.. code-block::

   yes | conda install -c conda-forge/label/cf202003 wxpython

Now you can install ``fsleyes`` using ``conda-forge``. ``fsleyes`` downloads a deprecated version of dcm2niix,
you can remove it, it will use your local version:

.. code-block::

   yes | conda install -c conda-forge fsleyes=0.34.2
   yes | conda remove --force dcm2niix

To check if this worked, you can run:

.. code-block::

    dcm2niix --version

It should show the latest version of ``dcm2niix``.

Using the Shimming Plugin for FSLeyes
=====================================

In the command line, type:

.. code-block::

   fsleyes

This will open the ``fsleyes`` application. Once in ``fsleyes``, click on
**File** > **Load plugin** and select **shimmingtoolbox/gui/st_plugin.py**.

It will ask if you want to install it permanently: say **Yes**. Then, quit and reopen ``fsleyes``.
The plugin should be accessible via **Settings** > **Ortho View 1** > **Shimming Toolbox**.
