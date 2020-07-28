# Installation

Before running this software you will need to install the following dependencies:
- MATLAB version 2019B or later
  - Optimization toolbox
  - Image processing toolbox
- [dcm2niix](https://github.com/rordenlab/dcm2niix#install)
- [SCT v 4.0.0](https://github.com/neuropoly/spinalcordtoolbox)

To install, download (or `git clone`) this repository and add this folder (with sub-folders) to the Matlab path.

Start Matlab via the Terminal in order to load the shell environment variable that will be needed to launch UNIX-based software (e.g. FSL Prelude).

!!! note
    For the command line start, MATLAB also needs to exist within the system path, e.g. For MacOS, add the following lines (adapted to refer to your version of MATLAB) to ~/.bash_profile

    export PATH=$PATH:/Applications/MATLAB_R2020a.app/bin/

<!-- TODO: update
Create the folder '~/Matlab/shimming/' and copy into it the contents [here](https://drive.google.com/open?id=15mZNpsuuNweMUO6H2iWdf5DxA4sQ_aYR)
-->

After Matlab has started, add update the environment to access all the functions by running:
~~~
cd <PATH_SHIMMINGTOOLBOX>
startup.m
~~~

<!--
*For phase unwrapping:*

To use the optional Abdul-Rahman 3D phase unwrapper, binaries must be compiled from the source code found in /external/source/ -->
