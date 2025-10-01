#!/usr/bin/env bash

set -e

temp_dir=$(pwd)
echo "Temporary directory created at: $temp_dir"
host=$(hostname)
host=${host//[0-9]/}
build_dir=${temp_dir}/build-laghos-${host}
mkdir -p ${build_dir}
installDir="/dev/shm/install"

start_test=$(date +'%s')

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
    --prune \
    --internalize \
    --num-trials 2 \
    --iterations 2 \
    --seed 0 \
    --no-specialize

  # Useful to start another run of Mneme to test more features (Mneme reuses previous runs etc)
  # Run with --specialize
  echo "Running Mneme with --specialize: $JSON_RECORD / $KERNEL_ID"
  mneme tune \
    -db ${JSON_RECORD} \
    -rid ${KERNEL_ID} \
    --db-dir ${DB_STORE} \
    --tuner-type optuna \
    --search-sampler QMCSampler \
    --prune \
    --internalize \
    --num-trials 2 \
    --iterations 3 \
    --seed 0 \
    --specialize

  duration=$SECONDS
  echo "Mneme optimization: $((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
}

if [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    ml load rocm/${MNEME_CI_ROCM_VERSION}
    export LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
    echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"
    export ROCM_ARCH=$(rocm_agent_enumerator | sed -n 2p)

    if [ -z "${ROCM_ARCH}" ]; then
        echo "ROCM_ARCH is not set or is empty"
        exit
    else
        echo "ROCM_ARCH = ${ROCM_ARCH}"
    fi

    # Mneme python package needs ROCm path not ${ROCM_PATH}/llvm
    export LLVM_INSTALL_DIR=${ROCM_PATH}
    # Instll Mneme python bindings
    if [[ ! -d "${installDir}/mneme-env" || ! -f "${installDir}/mneme-env/bin/activate" ]]; then
      python3 -m venv ${installDir}/mneme-env
      echo "Created virtual env ${installDir}/mneme-env"
    fi
    source ${installDir}/mneme-env/bin/activate
    echo "activated virtual env ${installDir}/mneme-env"

    export MNEME_ENABLE_DEBUG=${MNEME_CI_ENABLE_DEBUG}

    pip install .
    mkdir -p ${installDir}/lib64
    echo "Copying Mneme libs to ${installDir}/lib64"
    cp -r build/lib/lib64/*.so ${installDir}/lib64

    # Testing python install
    python3 -c "import mneme; print(mneme.__path__)"

    if [ $? -ne 0 ]; then
        echo "The python install seems broken."
        exit 1
    fi

    # Laghos is only working with HIP system for now
    echo "### BUILDING Laghos ###"
    build_laghos ${installDir}
    echo "### TESTING Laghos ###"

    OUTPUT_DIR="run-mneme"
    run_laghos ${installDir} ${OUTPUT_DIR}
    # Find the JSON file containing the kernel
    # We select one JSON file
    JSON_RECORD=$(find run-mneme/ -maxdepth 1 -name '*.json' -printf '%f' -quit)
    KERNEL_ID=$(jq -r '.instances | keys[]' ${OUTPUT_DIR}/${JSON_RECORD} | head -n 1)
    DB_STORE="mneme-result"
    run_mneme_laghos ${OUTPUT_DIR}/${JSON_RECORD} ${KERNEL_ID} ${DB_STORE}

else
    echo "$SYS_TYPE is not supported by that script"
    exit 1
fi

end_test=$(($(date +'%s') - $start_test))
echo "Build and tests for Laghos took $end_test seconds"
