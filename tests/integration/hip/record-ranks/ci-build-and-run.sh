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
    shift 2
    local data_base
    data_base=$(mktemp -d)

    # Capture rank-0 stderr so we can grep for the disclosure line.
    local err_log
    err_log=$(mktemp)

    LD_PRELOAD="${INSTALL_PREFIX}/lib64/librecord.so:${INSTALL_PREFIX}/lib64/libproteus.so" \
    MNEME_DATA_DIR_BASE="${data_base}" \
    "$@" \
    run_mpi "${nranks}" ./build/record_ranks 2>"${err_log}"

    cat "${err_log}" >&2
    echo "=> PASSED record-ranks (${label})"

    # Stash the err log path for callers that want to grep it.
    LAST_ERR_LOG="${err_log}"
    LAST_DATA_BASE="${data_base}"
}

# Scenario 1: default policy, multi-rank — expect rank 0 only + disclosure on stderr.
run_scenario "default-policy-4rank" 4
if ! grep -F "[mneme] Multi-rank run detected (4 ranks). Recording on rank 0 only." "${LAST_ERR_LOG}" >/dev/null; then
    echo "FAIL: default-policy disclosure line not found on stderr"
    exit 1
fi
echo "=> PASSED record-ranks (default-policy-disclosure-grep)"
rm -rf "${LAST_DATA_BASE}" "${LAST_ERR_LOG}"

# Scenario 2: explicit all — every rank records.
run_scenario "explicit-all-4rank" 4 env MNEME_RECORD_RANKS=all
rm -rf "${LAST_DATA_BASE}" "${LAST_ERR_LOG}"

# Scenario 3: explicit subset — ranks 0 and 2 record.
run_scenario "subset-0,2-4rank" 4 env MNEME_RECORD_RANKS=0,2
rm -rf "${LAST_DATA_BASE}" "${LAST_ERR_LOG}"

# Scenario 4: single-rank — default policy treats no-rank as single-process and records.
run_scenario "single-rank" 1
rm -rf "${LAST_DATA_BASE}" "${LAST_ERR_LOG}"

echo "ALL PASSED"
