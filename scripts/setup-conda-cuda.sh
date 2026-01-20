#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gitlab/mneme-common-ci.sh"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <conda-dir> <LLLVM-version> <python-version>" >&2
  exit 1
fi

CONDA_DIR=$1
LLVM_VERSION=$2
PYTHON_VERSION=$3


setup_conda_env $CONDA_DIR $LLVM_VERSION $PYTHON_VERSION 

