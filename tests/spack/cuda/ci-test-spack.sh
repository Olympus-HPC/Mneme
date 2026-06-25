#!/bin/bash

set -e

ml load cuda/${MNEME_CI_CUDA_VERSION}

# Install spack.
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_USER_CACHE_PATH=/tmp/spack-${CI_JOB_ID}/user_cache
git clone --quiet --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git /tmp/spack-${CI_JOB_ID}
source /tmp/spack-${CI_JOB_ID}/share/spack/setup-env.sh

# Create environment.
spack env create -d /tmp/mneme-spack-env-${CI_JOB_ID}
spack env activate /tmp/mneme-spack-env-${CI_JOB_ID}

spack external find
spack external find cuda

# mneme requires an llvm installation with +link_llvm_dylib enabled
# none of the LC nvidia systems provide it so we must build it ourselves
# because we'll use the clang built below to compile mneme, the compiler must be concrete
# in spack's eyes, so it must be concretized first and then mneme should be concretized
# (sans -f because they cannot be concretized together)
spack add llvm@${MNEME_CI_LLVM_VERSION} +clang +link_llvm_dylib targets=all
spack concretize -f

# Add repo and package.
PROTEUS_VERSION=$(cat ${CI_PROJECT_DIR}/PROTEUS_VERSION)
git clone --quiet --depth=1 --branch=cmelone/spack-updates git@github.com:cmelone/proteus.git /tmp/proteus-${CI_JOB_ID}
spack repo add /tmp/proteus-${CI_JOB_ID}/packaging/spack/spack_repo/proteus
spack repo add ${CI_PROJECT_DIR}/packaging/spack/spack_repo/mneme
spack add mneme@git.${CI_COMMIT_SHA} ~python +cuda cuda_arch=${MNEME_CI_CUDA_ARCH} ^cuda@${MNEME_CI_CUDA_VERSION} ^llvm@${MNEME_CI_LLVM_VERSION}

# Concretize and install.
spack concretize
spack install -v

# Cleanup.
rm -rf ${SPACK_USER_CACHE_PATH}
rm -rf /tmp/mneme-spack-env-${CI_JOB_ID}
rm -rf /tmp/spack-${CI_JOB_ID}
rm -rf /tmp/proteus-${CI_JOB_ID}
