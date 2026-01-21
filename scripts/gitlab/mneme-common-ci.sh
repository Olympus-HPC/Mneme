#!/usr/bin/env bash

setup_conda_env(){
  MINICONDA_DIR=$1
  LLVM_VERSION=$2
  PYTHON_VERSION=$3
  echo "Requested python version ${PYTHON_VERSION}"
  echo "Requested LLVM VERSION ${LLVM_VERSION}"
  echo "MINICONDA_DIR is ${MINICONDA_DIR}"
  if [[ ! -d ${MINICONDA_DIR} ]]; then
    mkdir -p ${MINICONDA_DIR}
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ${MINICONDA_DIR}/miniconda.sh
    bash ${MINICONDA_DIR}/miniconda.sh -b -u -p ${MINICONDA_DIR}
    rm ${MINICONDA_DIR}/miniconda.sh
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
    conda activate base
    conda create -y -n mneme -c conda-forge \
      python=${PYTHON_VERSION} clang=${LLVM_VERSION} clangxx=${LLVM_VERSION} \
      clangdev=${LLVM_VERSION} llvmdev=${LLVM_VERSION} lit=${LLVM_VERSION} \
      gcc=12 gxx=12
  else
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  fi
  conda activate mneme
}
