Welcome to shimming-toolbox!
============================

.. figure:: ./_static/shimming_toolbox_logo.png
   :alt: logo

.. NOTE ::
    This website is a work in progress.

``shimming-toolbox`` is an open-source Python software package enabling
a variety of MRI shimming (magnetic field homogenization) techniques
such as
`static <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.25587>`__
and `real-time <https://doi.org/10.1002/mrm.27089>`__ shimming for use
with standard manufacturer-supplied gradient/shim coils or with custom
"multi-coil" arrays. The toolbox provides useful set of command line tools as
well as a `fsleyes plugin <https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox>`__
dedicated to make shimming more accessible and more reproducible.

Insert image of the overview

Features
________

* Convert Dicoms to NIfTI
* Process phase images into fieldmaps

  * Available unwrappers: Prelude

* Create masks: Geometric, `SCT <https://spinalcordtoolbox.com/en/latest/>`__
* Perform shimming using different techniques: SH, multi-coil and gradient shimming

Installation
____________

See the :ref:`installation` page.

Usage
_____

Add usage



.. toctree::
   :hidden:
   :maxdepth: 4
   :glob:
   :caption: Contents:

   overview/introduction.rst
   overview/flowchart.rst
   getting_started/installation.rst
   getting_started/getting_started.rst
   getting_started/help.rst
   contributing/*
   about/*
   other-resources/hardware/*
   cli_reference/cli.rst
   api_reference/api.rst
