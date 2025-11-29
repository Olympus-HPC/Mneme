#!/bin/bash

set -e

if [[ -n "$CODECOV_TOKEN" ]]; then
  echo "CODECOV_TOKEN is set"
else
  echo "CODECOV_TOKEN is not set"
fi

mneme_src=$(pwd)
ml load python/${MNEME_CI_PYTHON_VERSION}
ml load rocm/${MNEME_CI_ROCM_VERSION}
export LLVM_INSTALL_DIR=${ROCM_PATH}/
test_dir="$TMP/mneme-ci-${CI_JOB_ID}"
echo "Test dir is ${test_dir}"
mkdir -p ${test_dir} 
pushd ${test_dir}
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install ${mneme_src}
pip install pytest pytest-cov
pytest -v -s ${mneme_src}/python/tests/ || exit $? 

# Upload to Codecov (only if token is available)
if [[ -n "$CODECOV_TOKEN" ]]; then
    echo "Uploading coverage to Codecov..."
    curl -s https://codecov.io/bash | bash -s -- -t "$CODECOV_TOKEN" -f coverage.xml || echo "Codecov upload failed"
else
    echo "No CODECOV_TOKEN set, skipping upload."
fi

