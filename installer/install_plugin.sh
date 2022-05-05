#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/utils.sh

set -e

ST_DIR=$HOME/shimming-toolbox
PYTHON_DIR=python
BIN_DIR=bin

# Define sh files
get_shell_rc_path

# Update PATH variables based on Shell type
DISPLAY_UPDATE_PATH="export PATH=\"$ST_DIR/$BIN_DIR:\$PATH\""

# Installation text to insert in shell config file
function edit_shellrc() {
  # Write text common to all shells
  if ! grep -Fq "FSLEYES_PLUGIN_SHIMMING_TOOLBOX (installed on" $RC_FILE_PATH; then
      (
        echo
        echo ""
        echo "# FSLEYES_PLUGIN_SHIMMING_TOOLBOX (installed on $(date +%Y-%m-%d\ %H:%M:%S))"
        echo "alias shimming-toolbox='bash $ST_DIR/$BIN_DIR/shimming-toolbox.sh'"
        echo ""
      ) >> "$RC_FILE_PATH"
      else
          echo "$RC_FILE_PATH file already updated from previous install, continuing to next step."
  fi
}

source $ST_DIR/$PYTHON_DIR/bin/activate

# Install fsleyes
print info "Installing fsleyes"
"$ST_DIR"/"$PYTHON_DIR"/bin/conda install -y -c conda-forge fsleyes=1.3.3

# Install fsleyes-plugin-shimming-toolbox
print info "Installing fsleyes-plugin-shimming-toolbox"
"$ST_DIR"/"$PYTHON_DIR"/bin/python -m pip install .

# Create launchers
print info "Creating launcher for fsleyes-plugin-shimming-toolbox..."
mkdir -p $ST_DIR/$BIN_DIR
chmod +x shimming-toolbox.sh
cp shimming-toolbox.sh $ST_DIR/$BIN_DIR/ # || die "Problem creating launchers!"

# Activate the launchers
export PATH=$ST_DIR/$BIN_DIR:$PATH

edit_shellrc
