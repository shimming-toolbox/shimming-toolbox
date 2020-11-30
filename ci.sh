#!/usr/bin/env bash

echo "Installation of dependencies"

# Install bleeding-edge dcm2niix
curl -JLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip
unzip -o dcm2niix_lnx.zip
sudo install dcm2nii* /usr/bin/

echo "shimming-toolbox installation"
pip install .
pip install coveralls

# Tests
echo "Download prelude"
st_download_data prelude
echo "Set up prelude"
prelude_path="$(pwd)"/prelude
chmod 500 "${prelude_path}"/prelude
PATH=${prelude_path}:${PATH}
export PATH
FSLOUTPUTTYPE=NIFTI_GZ
export FSLOUTPUTTYPE
echo "Launch general integrity test"
py.test . -v --cov shimmingtoolbox/ --cov-report term-missing

# Fail if any .sh files have warnings
echo "Check Bash scripts"
if [[ -n "$(ls examples/*.sh)" ]]; then shellcheck examples/*.sh; fi
