#include "DeviceTraits.hpp"

__global__ void compareDevBlobs(const char *Blob1, const char *Blob2,
                                bool Differences, uint64_t MemSize) {
  size_t ID = threadIdx.x + blockDim.x * blockIdx.x;
  size_t GridSize = blockDim.x * gridDim.x;
  bool Same = true;

  for (int id = ID; id < MemSize; ID += GridSize) {
    Same = Same && (Blob1[id] == Blob2[id]);
    if (!Same)
      break;
  }
  __syncthreads();
}
