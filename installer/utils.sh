#!/usr/bin/env bash

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

# Gets the shell rc file path based on the default shell.
# @output: THE_RC and RC_FILE_PATH vars are modified
function get_shell_rc_path() {
  if [[ "$SHELL" == *"bash"* ]]; then
    THE_RC="bash"
    RC_FILE_PATH="$HOME/.bashrc"
  elif [[ "$SHELL" == *"/sh"* ]]; then
    THE_RC="bash"
    RC_FILE_PATH="$HOME/.bashrc"
  elif [[ "$SHELL" == *"zsh"* ]]; then
    THE_RC="bash"
    RC_FILE_PATH="$HOME/.zshrc"
  elif [[ "$SHELL" == *"csh"* ]]; then
    THE_RC="csh"
    RC_FILE_PATH="$HOME/.cshrc"
  else
    find ~/.* -maxdepth 0 -type f
    die "ERROR: Shell was not recognized: $SHELL"
  fi
}
