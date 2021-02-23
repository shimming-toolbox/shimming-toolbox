# shimming-toolbox

 [![Build Status](https://github.com/shimming-toolbox/shimming-toolbox/workflows/CI-Tests/badge.svg?branch=master)](https://github.com/shimming-toolbox/shimming-toolbox/actions?query=workflow%3ACI-Tests+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/shimming-toolbox/shimming-toolbox/badge.svg?branch=master)](https://coveralls.io/github/shimming-toolbox/shimming-toolbox?branch=master) [![Documentation Status](https://readthedocs.org/projects/shimming-toolbox-py/badge/?version=latest)](https://shimming-toolbox.org/en/latest/)
 [![Twitter Follow](https://img.shields.io/twitter/follow/shimmingtoolbox.svg?style=social&label=Follow)](https://twitter.com/shimmingtoolbox)


Code for performing real-time shimming using external MRI shim coils

## Installation

The first dependency you need is [`dcm2niix`](https://github.com/rordenlab/dcm2niix) >= v1.0.20201102.
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

Please see the documentation for instructions on building and testing a [development version](https://shimming-toolbox.org/en/latest/getting_started/installation.html) of shimming-toolbox.

## Contributing

Please see our [contribution guidelines](https://shimming-toolbox.org/en/latest/contributing/CONTRIBUTING.html).
