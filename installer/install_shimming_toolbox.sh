#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/utils.sh"

set -e

ST_DIR=$HOME/shimming-toolbox

cd $ST_DIR

# Remove previous install
rm -rf "${ST_DIR}/shimming-toolbox"

print info "Downloading Shimming-Toolbox"

ST_VERSION="2d032209a0fc07ca3ac0741da96a844d03506cfa"

curl -L "https://github.com/shimming-toolbox/shimming-toolbox/archive/${ST_VERSION}.zip" > "shimming-toolbox-${ST_VERSION}.zip"

# Froze a commit, we can select a release later
# gunzip -c "shimming-toolbox-${ST_VERSION}.tar.gz" | tar xopf -
# unzip for now, when we use releases we can use the tar and gunzip
unzip -o "shimming-toolbox-${ST_VERSION}.zip"

# Rename the download to shimming-toolbox. This removes the hash from the folder name
mv "shimming-toolbox-${ST_VERSION}" "shimming-toolbox"

cd shimming-toolbox
make install CLEAN=false

# Copy coil config file in shimming toolbox directory
cp "${ST_DIR}/shimming-toolbox/config/coil_config.json" "${ST_DIR}/coil_config.json"

print info "To launch the plugin, load the environment variables then run:"
print list "shimming-toolbox"
