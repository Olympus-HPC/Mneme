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
mkdir -p ${test_dir}
echo "Test dir is ${test_dir}"
VENV_NAME="/usr/workspace/LExperts/ci/gitlab/venv-${LCSCHEDCLUSTER}-${MNEME_CI_ROCM_VERSION}-${MNEME_CI_PYTHON_VERSION}/"
mkdir -p ${VENV_NAME}
pushd ${test_dir}
python -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
python -m pip uninstall -y mneme
rm -rf ${VENV_NAME}/lib*/python*/site-packages/mneme
python -m pip install ${mneme_src}
python -m pip install pytest pytest-cov
pytest -v -s ${mneme_src}/python/tests/ || exit $? 
pushd ${mneme_src}
pytest --cov-report=xml:coverage-${CI_JOB_ID}.xml --cov-config=.coveragerc python/tests/

# Upload to Codecov (only if token is available)
if [[ -n "$CODECOV_TOKEN" ]]; then
    echo "Uploading coverage to Codecov..."
    echo "SHA is $CI_COMMIT_SHA"
    echo "Branch is $CI_COMMIT_BRANCH"
    export NODE_TLS_REJECT_UNAUTHORIZED=0

    curl -k -Os https://uploader.codecov.io/latest/linux/codecov
    chmod +x codecov
    ./codecov \
        -t "$CODECOV_TOKEN" \
        -f coverage-${CI_JOB_ID}.xml \
        -C "$CI_COMMIT_SHA" \
        -B "$CI_COMMIT_BRANCH" \
        --insecure \
        --disable-ci-detection \
        --slug=gh/Olympus-HPC/Mneme || echo "Codecov upload failed"
else
    echo "No CODECOV_TOKEN set, skipping upload."
fi

rm -f coverage-${CI_JOB_ID}.xml
deactivate
