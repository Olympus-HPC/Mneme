#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecordCUDA.hpp"
#include <cuda_runtime.h>

using namespace mneme;

class MnemeRecorderCUDAPreload : public MnemeRecorderCUDA {
private:
  static constexpr bool hasFatBinEnd = false;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDA &) = delete;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDA &&) = delete;

public:
  static MnemeRecorderCUDAPreload &instance() {
    static MnemeRecorderCUDAPreload Recorder{};
    return Recorder;
  }
};

extern "C" {
void __cudaRegisterFatBinaryEnd(void *ptr) {
  LOG_DEBUG("Entering mneme to finalize fatbinary");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  mneme.registerFatBinEnd(ptr);
}

void **__cudaRegisterFatBinary(void *fatbin) {
  LOG_DEBUG("Entering mneme to register fatbinary {}", fatbin);
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.registerFatBin(static_cast<FatBinaryWrapper_t *>(fatbin));
}

void __cudaRegisterVar(void **fatbinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size,
                       int constant, int global) {
  LOG_DEBUG("Entering Mneme to Register Variable");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  mneme.registerVar(fatbinHandle, hostVar, deviceAddress, deviceName, ext, size,
                    constant, global);
};

void __cudaRegisterFunction(void **fatbinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  LOG_DEBUG("Entering Mneme to Register function");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  mneme.registerFunc(fatbinHandle, hostFun, deviceFun, deviceName, thread_limit,
                     tid, bid, bDim, gDim, wSize);
};

cudaError_t cudaMalloc(void **ptr, size_t size) {
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtMalloc(ptr, size);
}

cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

cudaError_t cudaHostMalloc(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

cudaError_t cudaFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to Free pointer");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtFree(ptr);
};

cudaError_t cudaHostFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to HostFree pointer");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtHostFree(ptr);
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  LOG_DEBUG("Entering Mneme to Launch Kernel");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
}
