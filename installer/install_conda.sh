#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/utils.sh

print info "Installing conda in $ST_DIR/$PYTHON_DIR"

rm -rf "$TMP_DIR"

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'TMP_DIR')"
ST_DIR="$HOME/shimming-toolbox"
PYTHON_DIR="python"

cd "$ST_DIR"

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CONDA_INSTALLER=Miniconda3-latest-MacOSX-x86_64.sh
elif [[ "$OSTYPE" == "cygwin" ]]; then
    # POSIX compatibility layer and Linux environment emulation for Windows
    echo "Invalid operating system"
    exit 1
elif [[ "$OSTYPE" == "msys" ]]; then
    # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
    echo "Invalid operating system"
    exit 1
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "Invalid operating system"
    exit 1
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "Invalid operating system"
    exit 1
else
    echo "Invalid operating system"
    exit 1
fi

CONDA_INSTALLER_URL=https://repo.anaconda.com/miniconda/$CONDA_INSTALLER

installConda() {
    curl --url $CONDA_INSTALLER_URL --output $TMP_DIR/$CONDA_INSTALLER
    run bash "$TMP_DIR/$CONDA_INSTALLER" -p "$ST_DIR/$PYTHON_DIR" -b -f
    # export PATH=$HOME/miniconda3/bin:$PATH
    # source $HOME/miniconda3/bin/activate
}

installConda

rm -rf "$TMP_DIR"
