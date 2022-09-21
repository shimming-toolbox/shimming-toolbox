#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ST_PACKAGE_DIR="$( dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/utils.sh"

set -e

ST_DIR=$HOME/shimming-toolbox
PYTHON_DIR=python
BIN_DIR=bin

print info "Beginning shimming-toolbox install in $ST_DIR/$PYTHON_DIR"


# Define sh files
get_shell_rc_path

# Update PATH variables based on Shell type
DISPLAY_UPDATE_PATH="export PATH=\"$ST_DIR/$BIN_DIR:\$PATH\""

# Installation text to insert in shell config file
function edit_shellrc() {
  # Write text common to all shells
  if ! grep -Fq "SHIMMINGTOOLBOX (installed on" $RC_FILE_PATH; then
      (
        echo
        echo ""
        echo "# SHIMMINGTOOLBOX (installed on $(date +%Y-%m-%d\ %H:%M:%S))"
        echo "$DISPLAY_UPDATE_PATH"
        echo "export ST_DIR=$ST_DIR"
        echo ""
      ) >> "$RC_FILE_PATH"
      else
          echo "$RC_FILE_PATH file already updated from previous install, continuing to next step."
  fi
}

source "$ST_DIR/$PYTHON_DIR/bin/activate"

print info "Installing dcm2niix"
"$ST_DIR"/"$PYTHON_DIR"/bin/mamba install -y -c conda-forge dcm2niix python=3.9

print info "Installing shimming-toolbox"
cd "$ST_PACKAGE_DIR"
cp "config/dcm2bids.json" "$ST_DIR/dcm2bids.json"
"$ST_DIR"/"$PYTHON_DIR"/bin/python -m pip install -e ".[docs,dev]"

# Create launchers for Python scripts
print info "Creating launchers for Python scripts. List of functions available:"
mkdir -p "$ST_DIR/$BIN_DIR"

for file in "$ST_DIR"/"$PYTHON_DIR"/bin/*st_*; do
  cp "$file" "$ST_DIR/$BIN_DIR/" # || die "Problem creating launchers!"
  print list "$file"
done

# Activate the launchers
export PATH="$ST_DIR/$BIN_DIR:$PATH"

edit_shellrc

print info "Open a new Terminal window to load environment variables, or run:"
print list "source $RC_FILE_PATH"
