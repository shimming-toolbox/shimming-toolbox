Welcome to shimming-toolbox!
============================

|badge-ci| |badge-coveralls| |badge-doc| |badge-twitter|

.. |badge-ci| image:: https://github.com/shimming-toolbox/shimming-toolbox/workflows/CI-Tests/badge.svg?
    :alt: GitHub Actions CI
    :target: https://github.com/shimming-toolbox/shimming-toolbox/actions?query=workflow%3ACI-Tests+branch%3Amaster

.. |badge-coveralls| image:: https://coveralls.io/repos/github/shimming-toolbox/shimming-toolbox/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://coveralls.io/github/shimming-toolbox/shimming-toolbox?branch=master

.. |badge-doc| image:: https://readthedocs.org/projects/shimming-toolbox-py/badge/?version=latest
    :alt: Documentation Status
    :target: https://shimming-toolbox.org/en/latest/

.. |badge-twitter| image:: https://img.shields.io/twitter/follow/shimmingtoolbox.svg?style=social&label=Follow
    :alt: Twitter Follow
    :target: https://twitter.com/shimmingtoolbox

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
* Create custom coil profiles
* Perform shimming using different techniques: SH, custom multi-coil and gradient shimming

Installation
____________

See the :ref:`installation` page.

Usage
_____

**1. Command line**

Shimming-Toolbox's primary way to be used is through the command line. For example:

.. code-block:: console

  $ st_prepare_fieldmap -help

  Usage: st_prepare_fieldmap [OPTIONS] PHASE...

  Creates fieldmap (in Hz) from phase images. This function accommodates multiple echoes (2 or more) and phase difference. This function also accommodates 4D phase inputs, where the 4th dimension represents the time, in case multiple fieldmaps are acquired across time for the purpose of real-time shimming experiments.

  phase: Input path of phase nifti file(s), in ascending order: echo1,
  echo2, etc.

  Options:
  -mag PATH             Input path of mag nifti file
  -unwrapper [prelude]  Algorithm for unwrapping
  -output PATH          Output filename for the fieldmap, supported types : '.nii', '.nii.gz'
  -mask PATH            Input path for a mask. Used for PRELUDE
  -threshold FLOAT      Threshold for masking. Used for: PRELUDE
  -h, --help            Show this message and exit.

**2. Multi-command pipeline**

To facilitate reproducibility, commands can be chained together in a pipeline using multiple Shimming Toolbox commands. An `example <https://github.com/shimming-toolbox/shimming-toolbox/blob/master/examples/demo_realtime_zshimming.sh>`__ script is provided.

**3. Graphical User Interface (FSLeyes)**

Shimming Toolbox provides a GUI via a FSLeyes plugin. See the `plugin's Github page <https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox>`__ for installation.


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
