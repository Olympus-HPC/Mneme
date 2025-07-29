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

void __register_fatbinary(void **Handle, void *FatbinWrapper,
                          const char *ModuleId) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  auto Wrapper = (FatBinaryWrapper_t *)FatbinWrapper;
  LOG_DEBUG("Mneme received explicit request to register binary with ID: {} "
            "HANDLE:{} WRAPPER: {}",
            ModuleId, (void *)Handle, FatbinWrapper);
  mneme.explicitRegisterFatBin(Handle, Wrapper, ModuleId);
}

void __register_linked_binary(void *FatbinWrapper, const char *ModuleId) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  auto Wrapper = (FatBinaryWrapper_t *)FatbinWrapper;
  LOG_DEBUG(
      "Mneme received explicit request to register linked binary with ID: "
      "{} WRAPPER : {}",
      ModuleId, FatbinWrapper);
  mneme.explicitRegisterPreLinkedBinary((FatBinaryWrapper_t *)Wrapper,
                                        ModuleId);
}

void __register_fatbinary_end(void **Handle) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Mneme received explicit request to finalize fatbinary "
            "registration: "
            "{}",
            (void *)Handle);
  mneme.explicitEndRegisterFatBinary(Handle);
}

// void __register_fatbinary(void **Handle, void *FatbinWrapper,
//                           const char *ModuleId) {
//   auto Wrapper = (FatBinaryWrapper_t *)FatbinWrapper;
//   LOG_DEBUG("Entering mneme to register fatbinary {} {} with version {}",
//             ModuleId, FatbinWrapper, Wrapper->Version);
//   FILE *out = std::fopen(ModuleId, "wb");
//   serializeFatbin(Wrapper, out);
//   std::fclose(out);
// }

// void __cudaUnregisterFatBinary(void **ptr) {
//   LOG_DEBUG("Entering mneme to unregister fatbinary");
//   auto &mneme = MnemeRecorderCUDAPreload::instance();
//   mneme.unregisterFatBinEnd(ptr);
// }

void __cudaRegisterVar(void **fatbinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size,
                       int constant, int global) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Register Variable");
  mneme.registerVar(fatbinHandle, hostVar, deviceAddress, deviceName, ext, size,
                    constant, global);
};

void __cudaRegisterFunction(void **fatbinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Register function");
  mneme.registerFunc(fatbinHandle, hostFun, deviceFun, deviceName, thread_limit,
                     tid, bid, bDim, gDim, wSize);
};

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

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Launch Kernel");
  return mneme.rtLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtSetDevice(deviceID);
}

cudaError_t cudaGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtGetDevice(deviceID);
}
}
