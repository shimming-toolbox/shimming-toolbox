**************
FSLeyes Plugin
**************

Installation
============

First, you will need to make sure ``shimming-toolbox`` is installed using a ``conda``
environment, as detailed here in the ``Installation`` section.

Next, make sure dcm2niix is installed to the latest version.
https://github.com/rordenlab/dcm2niix

Then navigate to the Shimming Toolbox folder and activate your Shimming Toolbox environment:

```
cd shimming-toolbox
conda activate shim_venv
```

Next, install ``wxPython`` using ``conda-forge``:

```
conda install -c conda-forge/label/cf202003 wxpython
```

Now you can install ``fsleyes`` using ``conda-forge``. ``fsleyes`` downloads a deprecated version of dcm2niix,
you can remove it, it will use your local version:

```
yes | conda install -c conda-forge fsleyes=0.33.1
conda remove --force dcm2niix
```

Using the Shimming Plugin for FSLeyes
=====================================

In the command line, type:

```
fsleyes
```

This will open the ``fsleyes`` application.
