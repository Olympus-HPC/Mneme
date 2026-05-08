#include "MnemeAnnotationRuntime.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <cuda_runtime.h>
#include <utility>

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

cudaError_t __proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                               void **KernelArgs, uint64_t ShmemSize,
                               void *Stream) {
  LOG_DEBUG("Enetering Mneme to launch kernel");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                              static_cast<cudaStream_t>(Stream));
}

bool mneme_set_metadata_for_ptr(const void *ptr, mneme::Metadata md) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.setMetadataForPointer(ptr, std::move(md));
}

bool mneme_get_metadata_for_ptr(const void *ptr, mneme::Metadata *md) {
  if (!md)
    return false;

  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.getMetadataForPointer(ptr, *md);
}

bool mneme_erase_metadata_for_ptr(const void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.eraseMetadataForPointer(ptr);
}

}
