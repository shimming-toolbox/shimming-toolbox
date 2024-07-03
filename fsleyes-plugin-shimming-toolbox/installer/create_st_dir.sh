#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/utils.sh"

ST_DIR="$HOME/shimming-toolbox"
PYTHON_DIR="python"

mkdir -p "$ST_DIR"
run rm -rf "$ST_DIR/$PYTHON_DIR"
run mkdir -p "$ST_DIR/$PYTHON_DIR"
