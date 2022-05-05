#!/usr/bin/env bash

set -e

ST_DIR=$HOME/shimming-toolbox
PYTHON_DIR=python

source $ST_DIR/$PYTHON_DIR/bin/activate

fsleyes &
