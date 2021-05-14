#!/usr/bin/env bash

set -e

VENV_ID=1267b18e73341ad94da34474
VENV=pst_venv_$VENV_ID
ST_DIR=$HOME/shimming_toolbox
PYTHON_DIR=python

# conda activate base
source $ST_DIR/$PYTHON_DIR/etc/profile.d/conda.sh
conda activate $VENV

fsleyes &
