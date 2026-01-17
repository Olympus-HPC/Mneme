// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <cuda_runtime.h>

#ifdef __GNUC__
#define __align__(n) __attribute__((aligned(n)))
#else
#define __align__(n) __declspec(align(n))
#endif

using namespace mneme;

class MnemeRecorderCUDAPreload
    : public MnemeRecorder<mneme::DeviceVendors::CUDA> {
private:
  static constexpr bool hasFatBinEnd = true;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDAPreload &) = delete;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDAPreload &&) = delete;
  MnemeRecorderCUDAPreload() {
    // NOTE: This is important to be called in the initializer. As it enforces
    // the initialization/de-initialization order.
    // FIXME: Fix de-init fiasco order in some proper way
    LOG_DEBUG("Initializing preloaded library");
  }

public:
  static MnemeRecorderCUDAPreload &instance() {
    static MnemeRecorderCUDAPreload Recorder{};
    return Recorder;
  }
};

extern "C" {
cudaError_t cudaMalloc(void **ptr, size_t size) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  return mneme.rtMalloc(ptr, size);
}

cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  return mneme.rtManagedMalloc(ptr, size, flags);
};

cudaError_t cudaHostMalloc(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  return mneme.rtHostMalloc(ptr, size, flags);
}

cudaError_t cudaFree(void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Free pointer");
  return mneme.rtFree(ptr);
};

cudaError_t cudaHostFree(void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to HostFree pointer");
  return mneme.rtHostFree(ptr);
}

cudaError_t cudaSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtSetDevice(deviceID);
}

cudaError_t cudaGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtGetDevice(deviceID);
}
}
