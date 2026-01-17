#!/bin/bash

# This script runs the Mneme examples workflow in CI.
# Simplified approach:
# 1. Setup Python Venv.
# 2. pip install . (builds Mneme, Proteus, SPDLOG).
# 3. Build only the example binaries using CMake.
# 4. Run Record & Tune.

set -e

# Setup Temporary Directories
temp_dir=$(mktemp -d)
echo "Temporary directory created at: $temp_dir"

host=$(hostname)
host=${host//[0-9]/}

mneme_orig_src=$(pwd)
installDir=${temp_dir}/install # Not strictly needed if pip installs to venv, but useful for cmake logic

# --- Environment Setup ---
if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
    ml load gcc/11.2.1
    ml load cmake/3.23
    ml load cuda/12.2

    # Conda Setup for Python & LLVM
    MINICONDA_DIR=miniconda3
    if [[ ! -d ${MINICONDA_DIR} ]]; then
    mkdir -p ${MINICONDA_DIR}
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ./${MINICONDA_DIR}/miniconda.sh
    bash ./${MINICONDA_DIR}/miniconda.sh -b -u -p ./${MINICONDA_DIR}
    rm ./${MINICONDA_DIR}/miniconda.sh
    source ./${MINICONDA_DIR}/bin/activate
    conda create -y -n mneme -c conda-forge \
        python=3.10 clang=18.1.8 clangxx=18.1.8 llvmdev=18.1.8 lit=18.1.8 clangdev=18.1.8
    else
    source ./${MINICONDA_DIR}/bin/activate
    fi
    conda activate mneme

    export LLVM_INSTALL_DIR=$(llvm-config --prefix)
    echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"
    
elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    ml load rocm/${MNEME_CI_ROCM_VERSION}
    ml load python/${MNEME_CI_PYTHON_VERSION}
    export LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
    echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"
    
    # Venv Setup
    LOCAL_VENV_NAME="${temp_dir}/venv/"
    python -m venv "${LOCAL_VENV_NAME}"
    source ${LOCAL_VENV_NAME}/bin/activate
fi

echo "Using Python: $(which python)"

# --- Install Mneme (Core + Python) ---
echo "Installing Mneme..."

# Copy source to clean temp dir for pip install (avoids cmake build tree conflicts)
mneme_pip_src=${temp_dir}/mneme_src_pip
mkdir -p ${mneme_pip_src}
rsync -a --delete --exclude 'build*' "${mneme_orig_src}/" "${mneme_pip_src}/"

python -m pip install -U pip setuptools wheel
python -m pip uninstall -y mneme
python -m pip install optuna
# Install verbose to catch errors. This invokes setup.py -> CMakeBuild
# which builds Proteus, SPDLOG, and Mneme library, installing them into the venv site-packages.
python -m pip install -v ${mneme_pip_src}

# Retrieve installation path for MnemeConfig.cmake
# It should be in <venv>/lib/pythonX.Y/site-packages/mneme/native/lib64/cmake/Mneme
# We can find it via python.
MNEME_INSTALL_DIR=$(python -c "import mneme; import os; print(os.path.join(os.path.dirname(mneme.__file__), 'native'))")
echo "Mneme installed at: ${MNEME_INSTALL_DIR}"


# --- Build Examples ---
echo "Building Examples..."
build_examples_dir=${temp_dir}/build-examples
mkdir -p ${build_examples_dir}
pushd ${build_examples_dir}

# We need to point CMake to the Mneme installation we just made.
# The `setup.py` installs libs to `mneme/native`.
# We need to set `Mneme_DIR`, `Proteus_DIR`, `SPDLOG_DIR` or use `CMAKE_PREFIX_PATH`.

cmake_args=(
    "-DCMAKE_PREFIX_PATH=${MNEME_INSTALL_DIR};${MNEME_INSTALL_DIR}/lib64/cmake"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=on"
    "${mneme_orig_src}/examples/hecbench"
)

if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
    cmake_args+=("-DWITH_MNEME_EXAMPLE_CUDA=On")
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=70")
    # For CUDA, we might need to be careful about compilers if they aren't auto-detected from modules
    cmake_args+=("-DCMAKE_CXX_COMPILER=$(which clang++)")
    cmake_args+=("-DCMAKE_C_COMPILER=$(which clang)")
    cmake_args+=("-DCMAKE_CUDA_COMPILER=$(which clang++)")
elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    cmake_args+=("-DWITH_MNEME_EXAMPLE_HIP=On")
    cmake_args+=("-DCMAKE_CXX_COMPILER=amdclang++")
    cmake_args+=("-DCMAKE_C_COMPILER=amdclang")
fi

cmake "${cmake_args[@]}"
make -j 10

popd # back to wherever


# --- Execute Examples Workflow ---

execute_example() {
    name=$1
    binary_path=$2
    record_args=$3
    tune_script=$4
    db_name="${name}_db"

    echo "### TESTING $name ###"
    pushd ${mneme_orig_src}
    
    rm -rf ${db_name}
    
    # RECORD
    echo "Recording..."
    # Ensure mneme CLI is in path (it is in venv/bin)
    mneme record -rdb ${db_name} -- ${binary_path} ${record_args}
    
    # EXTRACT ID
    json_path=$(ls ${db_name}/*.json | head -n 1)
    record_id=$(python3 -c "import json; print(list(json.load(open('${json_path}'))['instances'].keys())[0])")
    
    # TUNE
    echo "Tuning..."
    ${tune_script} --record-db ${json_path} --record-id ${record_id}
    
    popd
    echo "### $name DONE ###"
}

# 1. WSM5
execute_example "wsm5" \
    "${build_examples_dir}/wsm5/wsm5-mneme" \
    "1" \
    "./examples/hecbench/wsm5/tune.py"

# 2. Bezier Surface
execute_example "bezier" \
    "${build_examples_dir}/bezier-surface/bezier-mneme" \
    "-f examples/hecbench/bezier-surface/input/control.txt -n 8192" \
    "./examples/hecbench/bezier-surface/tune.py"

# 3. MiniFE
execute_example "miniFE" \
    "${build_examples_dir}/miniFE/miniFE-mneme" \
    "-nx 220 -ny 200 -nz 190" \
    "./examples/hecbench/miniFE/tune.py"

# Clean up
rm -rf ${temp_dir}
