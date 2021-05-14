#!/usr/bin/env bash

VENV_ID=1267b18e73341ad94da34474
VENV=pst_venv_$VENV_ID

rm -rf "$TMP_DIR"

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'TMP_DIR')"
ST_DIR="$HOME/shimming_toolbox"
PYTHON_DIR="python"

# Print with color
# @input1: {info, code, error}: type of text
# rest of inputs: text to print
function print() {
  type=$1; shift
  case "$type" in
  # Display useful info (green)
  info)
    echo -e "\n\033[0;32m${*}\033[0m\n"
    ;;
  # To interact with user (no carriage return) (light green)
  question)
    echo -e -n "\n\033[0;92m${*}\033[0m"
    ;;
  # To display code that is being run in the Terminal (blue)
  code)
    echo -e "\n\033[0;34m${*}\033[0m\n"
    ;;
  # Warning message (yellow)
  warning)
    echo -e "\n\033[0;93m${*}\033[0m\n"
    ;;
  # Error message (red)
  error)
    echo -e "\n\033[0;31m${*}\033[0m\n"
    ;;
  esac
}

# Elegant exit with colored message
function die() {
  print error "$1"
  exit 1
}

# Run a command and display it in color. Exit if error.
# @input: string: command to run
function run() {
  ( # this subshell means the 'die' only kills this function and not the whole script;
    # the caller can decide what to do instead (but with set -e that usually means terminating the whole script)
    print code "$@"
    if ! "$@" ; then
      die "ERROR: Command failed."
    fi
  )
}

mkdir -p "$ST_DIR"
run rm -rf "$ST_DIR/$PYTHON_DIR"
run mkdir -p "$ST_DIR/$PYTHON_DIR"
cd "$ST_DIR"

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    apt-get install -y curl
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

# create py3.6 venv (for Keras/TF compatibility with Centos7, see issue #2270)
python/bin/conda create -y -n $VENV python=3.7

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
sed_i 's/^ENABLE_USER_SITE.*$/ENABLE_USER_SITE = False/' "$ST_DIR/$PYTHON_DIR/envs/$VENV/lib/python"*"/site.py"

# activate miniconda
# shellcheck disable=SC1091
source python/etc/profile.d/conda.sh

# set +u #disable safeties, for conda is not written to their standard.
conda activate $VENV
# set -u # reactivate safeties

rm -rf "$TMP_DIR"
