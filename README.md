# shimming-toolbox-py
[![Build Status](https://travis-ci.com/shimming-toolbox/shimming-toolbox-py.svg?branch=master)](https://travis-ci.com/shimming-toolbox/shimming-toolbox-py)

Code for performing real-time shimming using external MRI shim coils

## Installation

```
$ pip install shimmingtoolbox@git+https://github.com/shimming-toolbox/shimming-toolbox-py
```

Depending on your system, you may need to prefix this command with `sudo` (for a global install),
or write it with `install --user` (for a user-local install) in order to be allowed to install it.

## Development

To set up a development environment, make sure python is installed, then run

```
$ pip install -e ".[testing]"
```

This will add the project folder to your python path and install the necessary dependencies to run tests.

To run tests manually, do

```
$ pytest
```

See https://docs.pytest.org/ for more options.
