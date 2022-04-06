#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/utils.sh

ST_DIR="$HOME/shimming-toolbox"
PYTHON_DIR=python
cd "$ST_DIR"

print info "Creating new virtual environment in $ST_DIR/$PYTHON_DIR"

if [ "$(uname)" = "Darwin" ]; then
  # macOS polyfills

  # Linux has `realpath` and `readlink -f`.
  # BSD has `readlink -f`.
  # macOS has neither: https://izziswift.com/how-can-i-get-the-behavior-of-gnus-readlink-f-on-a-mac/
  # even though it *has* the function in its libc: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/realpath.3.html
  # Someone even wrote a whole C program just for this: https://github.com/harto/realpath-osx/.
  # https://stackoverflow.com/a/3572105 suggests some bash trickery but it
  # seems pretty fragile and that it probably doesn't actually expand symlinks.
  # So here, solve it with python. It's not great either, but since much of
  # this script already is just calling python at least it's reliable.
  (command -v realpath >/dev/null) || realpath() {
    python3 -c 'import sys, os; [print(os.path.realpath(f)) for f in sys.argv[1:]]' "$@"
  }


  # glue over the difference between how BSD and GNU sed work with -i
  # https://code-examples.net/en/q/56e314
  sed_i() {
    sed -i '' "$@"
  }
else
  sed_i() {
    sed -i "$@"
  }
fi

# make sure that there is no conflict with local python install by making $VENV an isolated environment.
# workaround for https://github.com/conda/conda/issues/7173
# see also
# * https://github.com/neuropoly/spinalcordtoolbox/pull/3067
# * https://github.com/neuropoly/spinalcordtoolbox/issues/3200
# this needs to be added very early in python's boot process
# so using sitecustomize.py or even just appending to the file are impossible.
sed_i 's/^ENABLE_USER_SITE.*$/ENABLE_USER_SITE = False/' "$ST_DIR/$PYTHON_DIR/lib/python"*"/site.py"

# activate miniconda
# shellcheck disable=SC1091
source python/etc/profile.d/conda.sh # useless?
