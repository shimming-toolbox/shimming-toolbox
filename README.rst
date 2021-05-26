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
well as a `fsleyes plugin <https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox#fsleyes-plugin-for-shimming-toolbox>`__
dedicated to make shimming more accessible and more reproducible.

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/overview/overview.gif
  :alt: Overview
  :width: 1000

Features
________

* Built-in DICOM to NIfTI conversion
* Masking tools for the brain and spinal cord
* Supports the creation and usage of a variety of coil profiles: Spherical harmonics, shim-only arrays, "AC-DC" multi-coil, etc.
* Supports different shimming scenarios: dynamic (slicewise), realtime (shim modulation with respiration), gradient z-shimming, two-region shimming (e.g., fat and brain)
* Powered by freely available software tools: `dcm2niix <https://github.com/rordenlab/dcm2niix>`__, `dcm2bids <https://github.com/UNFmontreal/Dcm2Bids>`__, `FSL-prelude <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#PRELUDE_.28phase_unwrapping.29>`__, `SCT <https://spinalcordtoolbox.com/en/latest/>`__, `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_.

Installation
____________

See the `Installing shimming-toolbox <https://shimming-toolbox.org/en/latest/getting_started/installation_st.html>`__ page.

Usage
_____

**1. Command Line**

The primary way to use ``shimming-toolbox`` is through the command line. For example:

.. code-block:: console

  $ st_prepare_fieldmap --help

  Usage: st_prepare_fieldmap [OPTIONS] PHASE...

  Creates fieldmap (in Hz) from phase images. This function accommodates multiple
  echoes (2 or more) and phase difference. This function also accommodates 4D phase
  inputs, where the 4th dimension represents the time, in case multiple fieldmaps are
  acquired across time for the purpose of real-time shimming experiments.

  phase: Input path of phase nifti file(s), in ascending order: echo1,
  echo2, etc.

  Options:
  -mag PATH             Input path of mag nifti file
  -unwrapper [prelude]  Algorithm for unwrapping
  -output PATH          Output filename for the fieldmap, supported types : '.nii', '.nii.gz'
  -mask PATH            Input path for a mask. Used for PRELUDE
  -threshold FLOAT      Threshold for masking. Used for: PRELUDE
  -h, --help            Show this message and exit.

.. admonition:: Note

  To facilitate reproducibility, commands can be chained together in a pipeline using multiple Shimming Toolbox commands. An `example <https://github.com/shimming-toolbox/shimming-toolbox/blob/master/examples/demo_realtime_zshimming.sh>`__ script is provided.

**2. Graphical User Interface (FSLeyes)**

``shimming-toolbox`` also features a graphical user interface (GUI) via a FSLeyes plugin. See the `plugin's installation page <https://shimming-toolbox.org/en/latest/getting_started/installation_gui.html>`__ for more information.

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/fsleyes_example.png
  :alt: Overview
  :width: 1000
