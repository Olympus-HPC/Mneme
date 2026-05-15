#!/bin/bash
set -e

# Run an MPI program with auto-detected scheduler (Flux or SLURM).
run_mpi() {
    local np=$1
    shift
    if [[ -n "$FLUX_JOB_ID" ]] || command -v flux &>/dev/null; then
        flux run -n$np "$@"
    elif [[ -n "$SLURM_JOB_ID" ]] || command -v srun &>/dev/null; then
        srun -n$np "$@"
    else
        echo "ERROR: No supported MPI launcher found (flux or slurm)"
        exit 1
    fi
}

# CI_PROJECT_DIR / TEST_DIR are normally set by the CI runner. Allow local use
# via positional defaults computed from the script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TEST_DIR:=${SCRIPT_DIR}}"
: "${CI_PROJECT_DIR:=$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Build Mneme into a temp install prefix.
INSTALL_PREFIX=${PWD}/install-mneme
cmake -S ${CI_PROJECT_DIR} -B build-mneme \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DMNEME_ENABLE_HIP=on \
    -DENABLE_TESTS=off

pushd build-mneme
make -j install
popd

# Build the test against the installed Mneme.
cmake -S ${TEST_DIR} -B build \
    -Dmneme_DIR="${INSTALL_PREFIX}/lib64/cmake/mneme" \
    -Dproteus_DIR="${INSTALL_PREFIX}/lib64/cmake/proteus"
pushd build
make -j
popd

run_scenario() {
    local label="$1"
    local nranks="$2"
    local expected_ranks="$3"
    local record_ranks="${4:-}"
    local data_base
    data_base=$(mktemp -d)

    (
        export LD_PRELOAD="${INSTALL_PREFIX}/lib64/librecord.so:${INSTALL_PREFIX}/lib64/libproteus.so"
        export MNEME_DATA_DIR_BASE="${data_base}"
        if [[ -n "${record_ranks}" ]]; then
            export MNEME_RECORD_RANKS="${record_ranks}"
        fi

        run_mpi "${nranks}" ./build/record_ranks "${expected_ranks}"
    )
    echo "=> PASSED record-ranks (${label})"

    LAST_DATA_BASE="${data_base}"
}

# Scenario 1: default policy, multi-rank — expect rank 0 only.
run_scenario "default-policy-4rank" 4 "0"
rm -rf "${LAST_DATA_BASE}"

# Scenario 2: explicit all — every rank records.
run_scenario "explicit-all-4rank" 4 "0,1,2,3" "all"
rm -rf "${LAST_DATA_BASE}"

# Scenario 3: explicit subset — ranks 0 and 2 record.
run_scenario "subset-0,2-4rank" 4 "0,2" "0,2"
rm -rf "${LAST_DATA_BASE}"

echo "ALL PASSED"
