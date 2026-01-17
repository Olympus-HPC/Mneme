#!/bin/bash

set -euo pipefail
trap 'echo "[ERROR] line $LINENO" >&2' ERR

log() { echo " #### [$(date +%T)] $* ###" >&2; }

log "CI job $CI_JOB_ID starting"

if [[ -n "$CODECOV_TOKEN" ]]; then
  log "CODECOV_TOKEN is set"
else
  log "CODECOV_TOKEN is not set"
fi

log "CI JOB ID IS ${CI_JOB_ID}"

test_dir="$(mktemp -d)"
mkdir -p ${test_dir}

# Make a copy of src to local FS to accelerate building etc. 
log "Start copying src to temp"
mneme_orig_src=$(pwd)
mkdir -p $test_dir/mneme_src/
rsync -a --delete "$mneme_orig_src" "$test_dir/mneme_src/"
mneme_src=$test_dir/mneme_src/Mneme
log "End copying mneme src to ${mneme_src} from ${mneme_orig_src}"

ml load python/${MNEME_CI_PYTHON_VERSION}
ml load rocm/${MNEME_CI_ROCM_VERSION}
export LLVM_INSTALL_DIR=${ROCM_PATH}/
log "Test dir is ${test_dir}"


# Make a copy of preinstalled deps to local FS for fast installation
#VENV_NAME="/usr/workspace/LExperts/ci/gitlab/venv-${LCSCHEDCLUSTER}-${MNEME_CI_ROCM_VERSION}-${MNEME_CI_PYTHON_VERSION}/"
#rm -rf ${VENV_NAME}/lib*/python*/site-packages/mneme

log "Starting making environment"
LOCAL_VENV_NAME="${test_dir}/venv/"
mkdir -p ${LOCAL_VENV_NAME}
python -m venv "${LOCAL_VENV_NAME}"
#rsync -a "$VENV_NAME/" "$LOCAL_VENV_NAME"
pushd ${test_dir}
source ${LOCAL_VENV_NAME}/bin/activate
echo "Environment is: ${LOCAL_VENV_NAME}"
echo "ENV will use python:"
which python

python -m pip uninstall -y mneme
log "Environment made"

log "Start installing mneme"
python -m pip  -v install ${mneme_src}
log "Finalized with installation"

log "Installing pytest"
python -m pip install pytest pytest-cov

log "Start running mneme tests"
pytest -v -s ${mneme_src}/python/tests/ || exit $? 
log "Done with testing"
pushd ${mneme_src}
pytest --cov-report=xml:coverage-${CI_JOB_ID}.xml --cov-config=.coveragerc python/tests/

# Upload to Codecov (only if token is available)
if [[ -n "$CODECOV_TOKEN" ]]; then
    log "Uploading coverage to Codecov..."
    log "SHA is $CI_COMMIT_SHA"
    log "Branch is $CI_COMMIT_BRANCH"
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


python -m pip uninstall -y mneme
python -m pip install -U pip setuptools wheel

# Try a editable install
log "Removed mneme"
log "Mneme src is ${mneme_src}"
log "Start editable install"
python -m pip install -e ${mneme_src}
# If everything is properly installed this command will not fail
log "End editable install"

mneme config cxx || exit $?

deactivate
rm -rf ${test_dir}
