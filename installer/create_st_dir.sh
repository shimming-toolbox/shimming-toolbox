#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/utils.sh

rm -rf "$TMP_DIR"

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'TMP_DIR')"
ST_DIR="$HOME/shimming-toolbox"
PYTHON_DIR="python"
BIN_DIR="bin"

mkdir -p "$ST_DIR"
run rm -rf "${ST_DIR}/${BIN_DIR}"
run rm -rf "${ST_DIR}/${PYTHON_DIR}"
run mkdir -p "${ST_DIR}/${PYTHON_DIR}"
