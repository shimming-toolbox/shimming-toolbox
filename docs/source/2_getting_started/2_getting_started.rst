Getting Started
===============

A good way to get started is looking at one of the example scripts.

Examples
--------

Example scripts are located under
`examples <https://github.com/shimming-toolbox/shimming-toolbox-py/tree/master/examples>`__.

general_demo
^^^^^^^^^^^^

This example shows the process of using the toolbox with dicom data and processing them to output an unwrapped phase
plot. More precisely, it will:

* Download unsorted dicoms
* Run dcm2bids to convert to nifti with bids structure
* Unwrap phase
* Save wrapped and unwrapped plot of first X,Y volume as unwrap_phase_plot.png in the output directory

To use the script, simply provide an output directory where the processing and output figure will be generated, if no
directory is provided, a directory *output_dir* will be created in the current directory.

::

    from examples.general_demo import general_demo
    general_demo()

