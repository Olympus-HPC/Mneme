#include "DeviceTraits.hpp"
#include <cstdint>

using namespace mneme;

#define FULL_MASK 0xffffffff

__device__ int warpReduceSum(int val) {
  // Perform butterfly reduction using __shfl_down_sync
  for (int offset = 32; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

__global__ void compareDevBlobs(const char *Blob1, const char *Blob2,
                                uint64_t *GSame, uint64_t MemSize) {
  size_t ID = threadIdx.x + blockDim.x * blockIdx.x;
  size_t GridSize = blockDim.x * gridDim.x;
  uint64_t NumEqual = 0;

  if (ID >= MemSize)
    return;

  for (int id = ID; id < MemSize; id += GridSize) {
    NumEqual += (uint64_t)(Blob1[id] == Blob2[id]);
  }
  atomicAdd(GSame, NumEqual);
}

bool DeviceTraits<DeviceVendors::HIP>::compareDeviceBlobs(const char *Blob1,
                                                          const char *Blob2,
                                                          uint64_t NumBytes) {
  uint64_t *NumEqualBytes;
  auto EC = DeviceTraits<HIP>::DeviceErrorCheck(DeviceTraits<HIP>::DeviceMalloc(
      reinterpret_cast<void **>(&NumEqualBytes), sizeof(uint64_t)));
  if (EC)
    FATAL_ERROR("Error in comparing blobs " + EC.value());

  EC = DeviceTraits<HIP>::DeviceErrorCheck(
      DeviceTraits<HIP>::DeviceMemset(NumEqualBytes, 0, sizeof(uint64_t)));
  if (EC)
    FATAL_ERROR("Error in comparing blobs " + EC.value());

  constexpr int NumThreads = 256;
  size_t NumBlocks = (NumBytes + NumThreads - 1) / NumThreads;
  compareDevBlobs<<<NumBlocks, NumThreads>>>(Blob1, Blob2, NumEqualBytes,
                                             NumBytes);

  EC = DeviceTraits<HIP>::DeviceErrorCheck(
      DeviceTraits<HIP>::DeviceSynchronize());
  if (EC)
    FATAL_ERROR("Error in comparing blobs " + EC.value());

  uint64_t HNumEqualBytes;
  EC = DeviceTraits<HIP>::DeviceErrorCheck(DeviceTraits<HIP>::DeviceCopy(
      &HNumEqualBytes, NumEqualBytes, sizeof(uint64_t),
      DeviceTraits<HIP>::MemcpyDeviceToHostKind()));
  if (EC)
    FATAL_ERROR("Error in comparing blobs " + EC.value());

  EC = DeviceTraits<HIP>::DeviceErrorCheck(
      DeviceTraits<HIP>::DeviceFree(NumEqualBytes));
  if (EC)
    FATAL_ERROR("Error in comparing blobs " + EC.value());

  DBG(Logger::logs("mneme") << "Totalbytes: " << NumBytes << " and "
                            << HNumEqualBytes << " are equal\n");

  return HNumEqualBytes == NumBytes;
}
