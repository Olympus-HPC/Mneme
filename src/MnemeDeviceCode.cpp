#include "DeviceTraits.hpp"
#include <cstdint>

using namespace mneme;

#define FULL_MASK 0xffffffff

__device__ int warpReduceSum(int val) {
  // Perform butterfly reduction using __shfl_down_sync
  for (int offset = 32; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

__global__ void compareDevBlobs(const char *Blob1, const char *Blob2,
                                uint64_t *same, uint64_t MemSize) {
  size_t ID = threadIdx.x + blockDim.x * blockIdx.x;
  size_t GridSize = blockDim.x * gridDim.x;
  bool Same = true;

  for (int id = ID; id < MemSize; ID += GridSize) {
    Same = Same && (Blob1[id] == Blob2[id]);
    if (!Same)
      break;
  }
  __syncthreads();
  int equals = warpReduceSum(Same);
  if (threadIdx.x == 0)
    atomicAdd(same, equals);
}

template <DeviceVendors Vendor>
bool compareDeviceBlobs(const char *Blob1, const char *Blob2,
                        uint64_t NumBytes) {

  uint64_t NumEqualBytes = 0;
  DeviceTraits<Vendor>::DeviceMalloc(&NumEqualBytes, sizeof(uint64_t));
  DeviceTraits<Vendor>::DeviceMemset(NumEqualBytes, 0, sizeof(uint64_t));
  constexpr int NumThreads = 256;

  return false;
}
