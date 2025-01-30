#!/bin/bash


host=$(hostname)
host=${host//[0-9]/}
rm -rf build_${host}
mkdir -p build-${host};
build_dir=build-${host}
installDir="$(pwd)/install"


build_proteus() {
  PROTEUS_ENABLE_HIP=$1
  PROTEUS_ENABLE_CUDA=$2
  PROTEUS_INSTALL_DIR=$3
  echo $PROTEUS_ENABLE_HIP $PROTEUS_ENABLE_CUDA

  LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
  mkdir build-proteus
  pushd build-proteus
  cmake .. \
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
}

build_spdlog() {
  SPDLOG_INSTALL_DIR=$1
  mkdir build-spdlog
  cd build-spdlog
  cmake \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=${SPDLOG_INSTALL_DIR} \
  .. 

  make -j 10
  make install -j 10
}

if [[ ! -d "proteus" ]]; then
  git clone  --branch mneme/integration git@github.com:Olympus-HPC/proteus.git
fi

if [[ ! -d "spdlog" ]]; then
  git clone https://github.com/gabime/spdlog.git
  pushd spdlog
  git checkout tags/v1.15.0 -b release/v1.15.0
  popd
fi


if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
ml load cuda/11.8
ml load gcc/11.2.1
ml load cmake/3.23
ml load python/3.8
#export PATH="/usr/workspace/LExperts/DaCe/usr_18.1.8/blueos_3_ppc64le_ib_p9/bin/:$PATH"
ml load clang/18.1.8-cuda-11.8.0-gcc-11.2.1

cmake .. \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DLLVM_INSTALL_DIR=$(dirname $(dirname $(which clang))) \
-DENABLE_CUDA=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ../
elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
ml load rocm/6.2

pushd proteus
build_proteus "ON" "OFF" $installDir
popd

pushd spdlog
build_spdlog $installDir
popd

pushd $build_dir 
cmake .. \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-Dproteus_Dir=$installDir \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=amdclang++ \
-DCMAKE_C_COMPILER=amdclang \
-DLLVM_INSTALL_DIR=$(dirname $(dirname $(which amdclang)))/llvm/ \
-DENABLE_HIP=On \
-DENABLE_DEBUG=On \
-DENABLE_TESTS=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ../
fi

make

popd
