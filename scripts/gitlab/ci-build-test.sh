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
  echo "Proteus: ENABLE_HIP: $PROTEUS_ENABLE_HIP ENABLE_CUDA: $PROTEUS_ENABLE_CUDA"
  mkdir build-proteus
  pushd build-proteus
  cmake .. \
  -DBUILD_SHARED=Off \
  -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  -DPROTEUS_ENABLE_HIP=${PROTEUS_ENABLE_HIP} \
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
  mkdir build-spdlog
  pushd build-spdlog
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
ml load python/3.8
#export PATH="/usr/workspace/LExperts/DaCe/usr_18.1.8/blueos_3_ppc64le_ib_p9/bin/:$PATH"
ml load clang/18.1.8-cuda-11.8.0-gcc-11.2.1
llvm_root=dirname $(dirname -- $(which clang))
echo "Setting root dir to be ${llvm_root}"
export LLVM_INSTALL_DIR=$(llvm-config --prefix)
ml load cuda/12.2

build_proteus "OFF" "ON" $installDir
echo "After proteus Current directory is $(pwd)"
build_spdlog $installDir
mneme_src=$(pwd)
pushd $build_dir 
cmake \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-Dproteus_DIR=$installDir \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DMNEME_ENABLE_HIP=Off \
-DMNEME_ENABLE_CUDA=On \
-DMNEME_ENABLE_DEBUG=${MNEME_CI_ENABLE_DEBUG} \
-DMNEME_ENABLE_TESTS=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ${mneme_src}



#cmake .. \
#-DCMAKE_BUILD_TYPE=Relwithdebinfo \
#-DCMAKE_INSTALL_PREFIX=$installDir \
#-DCMAKE_CXX_COMPILER=clang++ \
#-DCMAKE_C_COMPILER=clang \
#-DCMAKE_CUDA_COMPILER=clang++ \
#-DCMAKE_C_COMPILER=clang \
#-DCMAKE_CUDA_COMPILER=clang++ \
#-DLLVM_INSTALL_DIR=$(dirname $(dirname $(which clang))) \
#-DENABLE_CUDA=On \
#-DCMAKE_EXPORT_COMPILE_COMMANDS=on ../



elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
ml load rocm/${MNEME_CI_ROCM_VERSION}
export LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"

build_proteus "ON" "OFF" $installDir
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
