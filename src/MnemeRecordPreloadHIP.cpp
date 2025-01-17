#include "MnemeRecordHIP.hpp"

using namespace mneme;

class MnemeRecorderHIPPreload : public MnemeRecorderHIP {
private:
  static constexpr bool hasFatBinEnd = false;
  MnemeRecorderHIPPreload(MnemeRecorderHIP &) = delete;
  MnemeRecorderHIPPreload(MnemeRecorderHIP &&) = delete;

public:
  static MnemeRecorderHIPPreload &instance() {
    static MnemeRecorderHIPPreload Recorder{};
    return Recorder;
  }
};

extern "C" {
void __hipRegisterFatBinaryEnd(void *ptr) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  mneme.registerFatBinEnd(ptr);
}

void **__hipRegisterFatBinary(void *fatbin) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.registerFatBin(static_cast<FatBinaryWrapper_t *>(fatbin));
}

void __hipRegisterVar(void **fatbinHandle, char *hostVar, char *deviceAddress,
                      const char *deviceName, int ext, size_t size,
                      int constant, int global) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  mneme.registerVar(fatbinHandle, hostVar, deviceAddress, deviceName, ext, size,
                    constant, global);
};

void __hipRegisterFunction(void **fatbinHandle, const char *hostFun,
                           char *deviceFun, const char *deviceName,
                           int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim,
                           dim3 *gDim, int *wSize) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  mneme.registerFunc(fatbinHandle, hostFun, deviceFun, deviceName, thread_limit,
                     tid, bid, bDim, gDim, wSize);
};

hipError_t hipMalloc(void **ptr, size_t size) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipFree(void *ptr) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipHostFree(void *ptr) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                           void **args, size_t sharedMem, hipStream_t stream) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
}
