#!/usr/bin/env bash

set -e

VENV=pst_venv
ST_DIR=$HOME/shimming-toolbox
PYTHON_DIR=python

# conda activate base
source $ST_DIR/$PYTHON_DIR/etc/profile.d/conda.sh
conda activate $VENV

fsleyes &
