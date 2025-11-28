#!/bin/bash

set -e

mneme_src=$(pwd)
ml load python/${MNEME_CI_PYTHON_VERSION}
ml load rocm/${MNEME_CI_ROCM_VERSION}
export LLVM_INSTALL_DIR=${ROCM_PATH}/
test_dir="$TMP/mneme-ci-${CI_JOB_ID}"
echo "Test dir is ${test_dir}"
mkdir -p ${test_dir} 
pushd ${test_dir}
python -m venv create venv
source venv/bin/activate
pip install ${mneme_src}
pytest -v -s ${mneme_src}/python/tests/ 

