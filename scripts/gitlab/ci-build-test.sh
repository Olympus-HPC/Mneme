#!/bin/bash

temp_dir=$(pwd) #$(mktemp -d)
echo "Temporary directory created at: $temp_dir"
host=$(hostname)
host=${host//[0-9]/}
mkdir -p ${temp_dir}/build-${host};
build_dir=${temp_dir}/build-${host}
installDir="/dev/shm/install"


build_proteus() {
  echo "Building PROTEUS"
  git clone --depth 1 --branch "features/mneme-integrations" git@github.com:Olympus-HPC/proteus.git
  pushd proteus 
  git checkout features/mneme-integrations
  PROTEUS_ENABLE_HIP=$1
  PROTEUS_ENABLE_CUDA=$2
  PROTEUS_INSTALL_DIR=$3
  LINK_SHARED_LLVM=$4
  echo "Proteus: ENABLE_HIP: $PROTEUS_ENABLE_HIP ENABLE_CUDA: $PROTEUS_ENABLE_CUDA"
  mkdir build-proteus-${host}
  pushd build-proteus-${host}
  cmake .. \
  -DBUILD_SHARED=Off \
  -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  -DPROTEUS_ENABLE_HIP=${PROTEUS_ENABLE_HIP} \
  -DPROTEUS_LINK_SHARED_LLVM=${LINK_SHARED_LLVM} \
  -DPROTEUS_ENABLE_CUDA=${PROTEUS_ENABLE_CUDA} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
  -DENABLE_TESTS=Off \
  -DCMAKE_INSTALL_PREFIX=${PROTEUS_INSTALL_DIR}
  make -j 10
  make install -j 10
  popd
  popd
}

build_spdlog() {
  echo "Building SPDLOG"
  git clone --depth 1 --branch v1.15.0  --single-branch https://github.com/gabime/spdlog.git
  pushd spdlog
  SPDLOG_INSTALL_DIR=$1
  mkdir build-spdlog-${host}
  pushd build-spdlog-${host}
  cmake \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=${SPDLOG_INSTALL_DIR} \
  .. 

  make -j 10
  make install -j 10
  popd
  popd
}

if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
ml load gcc/11.2.1
ml load cmake/3.23
ml load cuda/12.2

# Install Clang/LLVM through conda.
MINICONDA_DIR=miniconda3
if [[ ! -d ${MINICONDA_DIR} ]]; then
mkdir -p ${MINICONDA_DIR}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ./${MINICONDA_DIR}/miniconda.sh
bash ./${MINICONDA_DIR}/miniconda.sh -b -u -p ./${MINICONDA_DIR}
rm ./${MINICONDA_DIR}/miniconda.sh
source ./${MINICONDA_DIR}/bin/activate
conda create -y -n mneme -c conda-forge \
    python=3.10 clang=18.1.8 clangxx=18.1.8 llvmdev=18.1.8 lit=18.1.8
else
source ./${MINICONDA_DIR}/bin/activate
fi
conda activate mneme 

LLVM_INSTALL_DIR=$(llvm-config --prefix)
export LLVM_INSTALL_DIR=$(llvm-config --prefix)
cpp=$(which clang++)
cc=$(which clang)
echo "Setting root dir to be ${LLVM_INSTALL_DIR}"

build_proteus "OFF" "ON" $installDir ON
echo "After proteus Current directory is $(pwd)"
build_spdlog $installDir
mneme_src=$(pwd)
set -x
pushd $build_dir 
echo "Current dir is $(pwd)"
cmake \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-Dproteus_DIR=$installDir \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=${cpp} \
-DCMAKE_C_COMPILER=${cc} \
-DMNEME_LINK_SHARED_LLVM=ON \
-DCMAKE_CUDA_COMPILER=${cpp} \
-DCMAKE_CUDA_ARCHITECTURES=70 \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DMNEME_ENABLE_HIP=Off \
-DMNEME_ENABLE_CUDA=On \
-DMNEME_ENABLE_DEBUG=${MNEME_CI_ENABLE_DEBUG} \
-DMNEME_ENABLE_TESTS=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ${mneme_src}


elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
ml load rocm/${MNEME_CI_ROCM_VERSION}
export LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"

build_proteus "ON" "OFF" $installDir OFF
echo "After proteus Current directory is $(pwd)"
build_spdlog $installDir
echo "After spdlog Current directory is $(pwd)"
mneme_src=$(pwd)
pushd $build_dir 
cmake \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-Dproteus_DIR=$installDir \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=amdclang++ \
-DCMAKE_C_COMPILER=amdclang \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DMNEME_ENABLE_HIP=On \
-DMNEME_ENABLE_DEBUG=${MNEME_CI_ENABLE_DEBUG} \
-DMNEME_ENABLE_TESTS=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ${mneme_src}
fi


make -j 10
echo "### TESTING ###"
ctest --output-on-failure
echo "### TESTING  ###"

make -j 10 install

popd
