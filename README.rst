Welcome to shimming-toolbox!
============================

|badge-releases| |badge-ci| |badge-coveralls| |badge-doc| |badge-twitter| |badge-license|

.. |badge-releases| image:: https://img.shields.io/github/v/release/shimming-toolbox/shimming-toolbox
    :alt: Releases
    :target: https://github.com/shimming-toolbox/shimming-toolbox/releases

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

.. |badge-license| image:: https://img.shields.io/github/license/shimming-toolbox/shimming-toolbox
   :alt: License
   :target: https://github.com/shimming-toolbox/shimming-toolbox/blob/master/LICENSE

``shimming-toolbox`` is an open-source Python software package enabling
a variety of MRI shimming (magnetic field homogenization) techniques
such as
`static <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.25587>`__
and `real-time <https://doi.org/10.1002/mrm.27089>`__ shimming for use
with standard manufacturer-supplied gradient/shim coils or with custom
"multi-coil" arrays. The toolbox provides useful set of command line tools as
well as a `fsleyes plugin <https://github.com/shimming-toolbox/fsleyes-plugin-shimming-toolbox#fsleyes-plugin-for-shimming-toolbox>`__
dedicated to make shimming more accessible and more reproducible.

For more details, see: `D'Astous A, Cereza G, Papp D, Gilbert KM, Stockmann JP, Alonso-Ortiz E, Cohen-Adad J. Shimming toolbox: An open-source software toolbox for B0 and B1 shimming in MRI. Magn Reson Med. 2022; 1-17. doi:10.1002/mrm.29528 <https://onlinelibrary.wiley.com/doi/10.1002/mrm.29528>`__

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/overview/overview.gif
  :alt: Overview
  :width: 1000

Features
________

* Built-in DICOM to NIfTI conversion
* Masking tools for the brain and spinal cord
* Create and use a variety of B0 shimming coil profiles: Spherical harmonics, shim-only arrays, "AC-DC" multi-coil, etc.
* Supports different B0 shimming scenarios: dynamic (slicewise), realtime (shim modulation with respiration), gradient z-shimming, two-region shimming (e.g., fat and brain)
* RF shimming for parallel transmit systems (a.k.a. B1+ shimming)
* Powered by freely available software tools: `dcm2niix <https://github.com/rordenlab/dcm2niix>`__, `dcm2bids <https://github.com/UNFmontreal/Dcm2Bids>`__, `FSL-prelude <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#PRELUDE_.28phase_unwrapping.29>`__, `SCT <https://spinalcordtoolbox.com/en/latest/>`__, `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_.

Installation
____________

See the `Installing shimming-toolbox <https://shimming-toolbox.org/en/latest/user_section/install.html>`__ page.

Usage
_____

**1. Graphical User Interface (FSLeyes)**

``shimming-toolbox`` features a graphical user interface (GUI) via a FSLeyes plugin.

.. figure:: https://raw.githubusercontent.com/shimming-toolbox/doc-figures/master/fsleyes/fsleyes_example.png
  :alt: Overview
  :width: 1000

**2. Command Line**

``shimming-toolbox`` can be used on the the command line. For example:

.. code-block:: console

  $ st_prepare_fieldmap -h

  Usage: st_prepare_fieldmap [OPTIONS] PHASE...

  Creates fieldmap (in Hz) from phase images.

  This function accommodates multiple echoes (2 or more) and phase difference.
  It also accommodates 4D phase inputs, where the 4th dimension represents the
  time, in case multiple field maps are acquired across time for the purpose
  of real-time shimming experiments. For non Siemens phase data, see
  --autoscale-phase option.

  PHASE: Input path of phase NIfTI file(s), in ascending order: echo1, echo2,
  etc. The BIDS metadata JSON file associated with each phase file is
  required, it will be fetched automatically using the same name as the NIfTI
  file.

  Example of use (Multiple echoes) : st_prepare_fieldmap phase_echo1.nii.gz
  phase_echo2.nii.gz phase_echo3.nii.gz --mag mag.nii.gz

  Example of use (Phase difference): st_prepare_fieldmap phasediff.nii.gz
  --mag mag.nii.gz

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
      --mask PATH                    Input path for a mask.
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

.. admonition:: Note

  To facilitate reproducibility, commands can be chained together in a pipeline using multiple Shimming Toolbox commands. An `example <https://github.com/shimming-toolbox/shimming-toolbox/blob/master/examples/demo_realtime_shimming.sh>`__ script is provided.

The different commands of Shimming Toolbox can be found in the `Command Line Tools page <https://shimming-toolbox.org/en/latest/cli_reference/cli.html>`__.
