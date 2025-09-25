#!/usr/bin/env bash

set -e

temp_dir=$(pwd) #$(mktemp -d)
echo "Temporary directory created at: $temp_dir"
host=$(hostname)
host=${host//[0-9]/}
build_dir=${temp_dir}/build-${host}
mkdir -p ${build_dir}
installDir="/dev/shm/install"

build_hypre(){
  LOCAL_DIR=$1
  hypre_version=$2
  if [ ! -d hypre ]; then
    git clone --depth 1 --branch $hypre_version https://github.com/hypre-space/hypre.git
  fi

  pushd hypre
  pushd src
  rocm_path=$(realpath $(dirname $(which hipcc))/../)
  rocm_mpi_path=$(realpath $(dirname $(which mpicc))/../)
  make distclean 2>/dev/null || true
  CUFLAGS="-fpass-plugin=${LOCAL_DIR}/lib64/libregdeviceir.so -O3 -std=c++14 -x hip --offload-arch=${ROCM_ARCH}" CC=mpicc CXX=mpicxx CXXFLAGS="std=c++17 -fPIC" CFLAGS="-fPIC" ./configure \
    --prefix=${LOCAL_DIR} \
    --with-extra-ldpath="${LOCAL_DIR}/lib64/" \
    --with-MPI-libs="mpi mpich mneme_shallow" \
    --with-MPI-lib-dirs=${rocm_mpi_path}/lib \
    --with-MPI-include=${rocm_mpi_path}/include \
    --enable-fortran \
    --with-hip

  make -j 4
  make check || true
  make install
  popd
  popd
}

build_metis(){
  LOCAL_DIR=$1
  currDir=$(pwd)
  if [ ! -d metis ]; then
    git clone --depth 1 https://github.com/mfem/tpls.git
    tar xzf tpls/metis-4.0.3.tar.gz
    mv metis-4.0.3 metis
    rm -rf tpls
  fi

  pushd metis
  # The Makefile in metis is broken
  sed -i 's/^CC = cc$/CC ?= cc/' Makefile.in
  CC=amdclang CXX=amdclang++ CPP=amdclang++ make -C Lib OPTFLAGS=-Wno-error=implicit-function-declaration
  cp libmetis.a $LOCAL_DIR/lib/
  popd
}

build_mfem(){
  LOCAL_DIR=$1
  mfem_version=$2
  if [[ ! -d "mfem" ]]; then
    git clone --branch ${mfem_version} --depth 1 https://github.com/mfem/mfem.git
  fi
  pushd mfem
  make distclean
  CXX=mpicxx make phip HIP_ARCH=${ROCM_ARCH} HIP_FLAGS="-fpass-plugin=${LOCAL_DIR}/lib64/libregdeviceir.so" METIS_DIR=$LOCAL_DIR/lib -j MPICXX=mpicxx HYPRE_OPT=-I${LOCAL_DIR}/include HYPRE_LIB=-L${LOCAL_DIR}/lib
  make -j 4
  make install
  popd
}

build_laghos() {
  LOCAL_DIR=$1

  echo "Building HYPRE"
  build_hypre ${LOCAL_DIR} v2.32.0
  echo "Building METIS"
  build_metis ${LOCAL_DIR}
  echo "Building MFEM"
  build_mfem ${LOCAL_DIR} v4.7

  echo "Building Laghos"
  if [[ ! -d "laghos" ]]; then
    git clone --depth 1 https://github.com/CEED/Laghos.git laghos
  fi
  pushd laghos
  # Laghos makefile needs some modifications to work properly with Mneme
  # sed -i 's|^MFEM_DIR ?= \.\./mfem$|MFEM_DIR ?= deps/mfem/|' makefile
  sed -i 's/^LAGHOS_LIBS = \$(MFEM_LIBS) \$(MFEM_EXT_LIBS)$/LAGHOS_LIBS = \$(MFEM_LIBS) \$(MFEM_EXT_LIBS) -lHYPRE -lrocsparse -lrocrand/' makefile
  sed -i 's/cd \$(<D); \$(CCC) -c \$(<F)/cd \$(<D); \$(CCC) -fgpu-rdc -c \$(<F)/' makefile
  sed -i 's/\$(MFEM_CXX) \$(MFEM_LINK_FLAGS) -o laghos/\$(MFEM_CXX) \$(MFEM_LINK_FLAGS) -fgpu-rdc --hip-link -o laghos/' makefile

  LDFLAGS=${LOCAL_DIR}/lib64/libmneme_shallow.so make -j4
  make install PREFIX=${LOCAL_DIR}
  popd
}

run_laghos() {
  INSTALL_DIR=$1
  OUTPUT_DIR="$2"

  if ! [ -x "$(command -v ${INSTALL_DIR}/laghos)" ]; then
    echo "Error: laghos is not installed."
    exit 1
  fi

  RS=3
  TF=0.0033
  CMD="${INSTALL_DIR}/laghos -p 1 -dim 2 -pa -tf ${TF} -d hip -rs ${RS}"

  export MNEME_LOG_LEVEL=critical
  export MNEME_PAGE_SIZE=16
  # export AMD_LOG_LEVEL=4

  mkdir -p $OUTPUT_DIR
  SECONDS=0
  env -C $OUTPUT_DIR LD_PRELOAD=${INSTALL_DIR}/lib64/librecord.so \
    LD_LIBRARY_PATH=/usr/tce/packages/cce/cce-18.0.1-magic/cce/x86_64/lib/:${INSTALL_DIR}/lib64/:$LD_LIBRARY_PATH \
    $CMD | tee output-laghos.log
  duration=$SECONDS
  echo "$OUTPUT_DIR: $((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
}

run_mneme_laghos() {
  JSON_RECORD="$1"
  KERNEL_ID="$2"
  DB_STORE="$3"
  export MNEME_LOG_LEVEL=critical

  SECONDS=0
  echo "Running Mneme without --specialize: $JSON_RECORD / $KERNEL_ID"

  mneme tune \
    -db ${JSON_RECORD} \
    -rid ${KERNEL_ID} \
    --db-dir ${DB_STORE} \
    --tuner-type optuna \
    --search-sampler QMCSampler \
    --suffix "ci" \
    --prune \
    --internalize \
    --num-trials 2 \
    --iterations 2 \
    --seed 0 \
    --no-specialize

  # Useful to start another run of Mneme to test more features (Mneme reuses previous runs etc)
  mneme tune \
    -db ${JSON_RECORD} \
    -rid ${KERNEL_ID} \
    --db-dir ${DB_STORE} \
    --tuner-type optuna \
    --search-sampler QMCSampler \
    --suffix "ci" \
    --prune \
    --internalize \
    --num-trials 2 \
    --iterations 4 \
    --seed 0 \
    --no-specialize

  duration=$SECONDS
  echo "Mneme optimization: $((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
}

build_proteus() {
  echo "Building PROTEUS"
  if [[ ! -d "proteus" ]]; then
    git clone --depth 1 https://github.com/Olympus-HPC/proteus.git
  fi
  pushd proteus 
  PROTEUS_ENABLE_HIP=$1
  PROTEUS_ENABLE_CUDA=$2
  PROTEUS_INSTALL_DIR=$3
  LINK_SHARED_LLVM=$4
  echo "Proteus: ENABLE_HIP: $PROTEUS_ENABLE_HIP ENABLE_CUDA: $PROTEUS_ENABLE_CUDA"
  mkdir -p build-proteus-${host}
  pushd build-proteus-${host}
  cmake .. \
  -DBUILD_SHARED=Off \
  -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
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
  if [[ ! -d "spdlog" ]]; then
    git clone --depth 1 --branch v1.15.0  --single-branch https://github.com/gabime/spdlog.git
  fi
  pushd spdlog
  SPDLOG_INSTALL_DIR=$1
  mkdir -p build-spdlog-${host}
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
    python=3.10 clang=18.1.8 clangxx=18.1.8 llvmdev=18.1.8 lit=18.1.8 clangdev=18.1.8
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
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
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
export ROCM_ARCH=$(rocm_agent_enumerator | sed -n 1p)

if [ -z "${ROCM_ARCH}" ]; then
  echo "ROCM_ARCH is not set or is empty"
  exit
else
  echo "ROCM_ARCH = ${ROCM_ARCH}"
fi

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

if [[ "${MNEME_CI_TEST_LAGHOS}" == "on" || "${MNEME_CI_TEST_LAGHOS}" == "On" || "${MNEME_CI_TEST_LAGHOS}" == "ON" ]]; then
  # Laghos is only working with HIP system for now
  echo "### BUILDING Laghos ###"
  build_laghos ${installDir}

  # Mneme python package needs ROCm path not ${ROCM_PATH}/llvm
  export LLVM_INSTALL_DIR=${ROCM_PATH}
  # Instll Mneme python bindings
  if [[ ! -d "${installDir}/mneme-env" || ! -f "${installDir}/mneme-env/bin/activate" ]]; then
    python3 -m venv ${installDir}/mneme-env
    echo "Created virtual env ${installDir}/mneme-env"
  fi
  source ${installDir}/mneme-env/bin/activate
  echo "activated virtual env ${installDir}/mneme-env"

  pip install .
  mkdir -p ${installDir}/lib64

  # Testing python install
  python3 -c "import mneme; print(mneme.__path__)"

  if [ $? -ne 0 ]; then
    echo "The python install seems broken."
    exit 1
  fi

  echo "### BUILDING Laghos ###"
  OUTPUT_DIR="run-mneme"
  run_laghos ${installDir} ${OUTPUT_DIR}
  echo "### TESTING Laghos ###"
  # Find the JSON file containing the kernel
  # We select one JSON file
  JSON_RECORD=$(find run-mneme/ -maxdepth 1 -name '*.json' -printf '%f' -quit)
  KERNEL_ID=$(jq -r '.instances | keys[]' ${OUTPUT_DIR}/${JSON_RECORD} | head -n 1)
  DB_STORE="mneme-result"
  run_mneme_laghos ${OUTPUT_DIR}/${JSON_RECORD} ${KERNEL_ID} ${DB_STORE}
fi
