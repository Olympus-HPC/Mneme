// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <hip/hip_runtime.h>

using namespace mneme;

class MnemeRecorderHIPPreload
    : public MnemeRecorder<mneme::DeviceVendors::HIP> {
private:
  static constexpr bool hasFatBinEnd = false;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &) = delete;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &&) = delete;

public:
  static MnemeRecorderHIPPreload &instance() {
    static MnemeRecorderHIPPreload Recorder{};
    return Recorder;
  }
};

extern "C" {
hipError_t hipMalloc(void **ptr, size_t size) {
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to Free pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipHostFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to HostFree pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtSetDevice(deviceID);
}

hipError_t hipGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtGetDevice(deviceID);
}

hipError_t __jit_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                               void **KernelArgs, uint64_t ShmemSize,
                               void *Stream) {
  LOG_DEBUG("Enetering Mneme to launch kernel");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                              static_cast<hipStream_t>(Stream));
}
}
