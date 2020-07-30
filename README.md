# shimming-toolbox-py
[![Build Status](https://travis-ci.com/shimming-toolbox/shimming-toolbox-py.svg?branch=master)](https://travis-ci.com/shimming-toolbox/shimming-toolbox-py)

Code for performing real-time shimming using external MRI shim coils

## Installation

The first dependency you need is [`dcm2niix`](https://github.com/rordenlab/dcm2niix) >= v1.0.20200331.
You can download precompiled binaries from https://github.com/rordenlab/dcm2niix/releases/latest -- make
sure to add them to your PATH.
It is also [in the AUR](https://aur.archlinux.org/packages/dcm2niix/) (`pikaur -Sy dcm2niix`),
[`brew`](https://github.com/Homebrew/homebrew-core/blob/master/Formula/dcm2niix.rb) (`brew install dcm2niix`),
and [`conda`](https://anaconda.org/conda-forge/dcm2niix) (`conda install -c conda-forge/label/cf202003 dcm2niix`).
Unfortunately, the version [currently in Debian/Ubuntu](https://packages.ubuntu.com/eoan/dcm2niix) (`apt install dcm2niix`) is too old to work reliably.

Once you have `dcm2niix`, install this package and the rest of its dependencies with:

```
$ pip install shimmingtoolbox@git+https://github.com/shimming-toolbox/shimming-toolbox-py
```

Depending on your system, you may need to prefix this command with `sudo` (for a global install),
or write it with `install --user` (for a user-local install) in order to be allowed to install it.

## Development

To set up a development environment, make sure python and `dcm2niix` are installed, then run

```
$ pip install -e ".[testing]"
```

This will add the project folder to your python path and install the necessary dependencies to run tests.

To run tests manually, do

```
$ pytest
```

See https://docs.pytest.org/ for more options.
