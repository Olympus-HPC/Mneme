#include "mneme/DeviceTraits.hpp"
#include <cstdint>
#include <cstring>
#include <mneme/MnemeLogger.hpp>

using namespace mneme;

#define FULL_MASK 0xffffffff

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using DeviceVendorTraits = DeviceTraits<DeviceVendors::CUDA>;
#endif

__global__ void compareDevBlobs(const char *Blob1, const char *Blob2,
                                unsigned long long *GSame, uint64_t MemSize) {
  size_t ID = threadIdx.x + blockDim.x * blockIdx.x;
  size_t GridSize = blockDim.x * gridDim.x;
  unsigned long long NumEqual = 0;

  if (ID >= MemSize)
    return;

  for (int id = ID; id < MemSize; id += GridSize) {
    NumEqual += (uint64_t)(Blob1[id] == Blob2[id]);
  }
  atomicAdd(GSame, NumEqual);
}

bool DeviceVendorTraits::compareDeviceBlobs(const char *Blob1,
                                            const char *Blob2,
                                            uint64_t NumBytes) {
  unsigned long long *NumEqualBytes;
  auto EC =
      DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceMalloc(
          reinterpret_cast<void **>(&NumEqualBytes),
          sizeof(unsigned long long)));
  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceMemset(
      NumEqualBytes, 0, sizeof(unsigned long long)));
  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  constexpr int NumThreads = 256;
  size_t NumBlocks = (NumBytes + NumThreads - 1) / NumThreads;
  compareDevBlobs<<<NumBlocks, NumThreads>>>(Blob1, Blob2, NumEqualBytes,
                                             NumBytes);

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());
  LOG_DEBUG("Comparing {} bytes located at Addr {} and Addr {}", NumBytes,
            (void *)Blob1, (void *)Blob2);

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  uint64_t HNumEqualBytes;
  EC = DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceCopy(
      &HNumEqualBytes, NumEqualBytes, sizeof(uint64_t),
      DeviceVendorTraits::MemcpyDeviceToHostKind()));

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceFree(NumEqualBytes));
  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  LOG_DEBUG("There are {} different bytes out of  {} total bytes",
            NumBytes - HNumEqualBytes, NumBytes);
  return HNumEqualBytes == NumBytes;
}
