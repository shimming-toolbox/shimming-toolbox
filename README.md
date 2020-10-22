# shimming-toolbox

[![Build Status](https://travis-ci.com/shimming-toolbox/shimming-toolbox.svg?branch=master)](https://travis-ci.com/shimming-toolbox/shimming-toolbox) [![Documentation Status](https://readthedocs.org/projects/shimming-toolbox/badge/?version=latest)](https://www.shimming-toolbox.org/en/latest/?badge=latest) [![Coverage Status](https://coveralls.io/repos/github/shimming-toolbox/shimming-toolbox/badge.svg?branch=master)](https://coveralls.io/github/shimming-toolbox/shimming-toolbox?branch=master)


Code for performing real-time shimming using external MRI shim coils

## Installation

The first dependency you need is [`dcm2niix`](https://github.com/rordenlab/dcm2niix) >= v1.0.20200331.
You can download precompiled binaries from https://github.com/rordenlab/dcm2niix/releases/latest -- make
sure to add them to your PATH.
It is also [in the AUR](https://aur.archlinux.org/packages/dcm2niix/) (`pikaur -Sy dcm2niix`),
[`brew`](https://github.com/Homebrew/homebrew-core/blob/master/Formula/dcm2niix.rb) (`brew install dcm2niix`),
and [`conda`](https://anaconda.org/conda-forge/dcm2niix) (`conda install -c conda-forge/label/cf202003 dcm2niix`).
Unfortunately, the version [currently in Debian/Ubuntu](https://packages.ubuntu.com/eoan/dcm2niix) (`apt install dcm2niix`) is too old to work reliably.

You will also need to install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) using Python 2.

Once you have `dcm2niix` and FSL, install this package and the rest of its dependencies with:

```
$ pip install shimmingtoolbox@git+https://github.com/shimming-toolbox/shimming-toolbox
```

Depending on your system, you may need to prefix this command with `sudo` (for a global install),
or write it with `install --user` (for a user-local install) in order to be allowed to install it.

## Development

Please see the documentation for instructions on building and testing a [development version](https://shimming-toolbox.org/en/latest/2_getting_started/1_installation.html) of shimming-toolbox.

## Contributing

Please see our [contribution guidelines](docs/source/3_contributing/CONTRIBUTING.rst).
