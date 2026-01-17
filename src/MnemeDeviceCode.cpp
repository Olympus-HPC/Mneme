// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void compareDevBlobs(const char *Blob1, const char *Blob2,
                                unsigned long long *GSame, uint64_t MemSize) {
  uint64_t tid = (uint64_t)threadIdx.x;
  uint64_t ID  = tid + (uint64_t)blockDim.x * (uint64_t)blockIdx.x;
  uint64_t GridSize = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

  unsigned long long local = 0;
  for (uint64_t i = ID; i < MemSize; i += GridSize) {
    local += (Blob1[i] == Blob2[i]);
  }

  __shared__ unsigned long long shmem[NUM_THREADS_PER_BLOCK];
  shmem[threadIdx.x] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shmem[threadIdx.x] += shmem[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(GSame, shmem[0]);
  }
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

  uint64_t NumBlocks = min( (NumBytes + NUM_THREADS_PER_BLOCK - 1)/NUM_THREADS_PER_BLOCK,
                  (uint64_t)8192 );   // DN -- you could also set this to something like SMs*20 to be more device dependent
  compareDevBlobs<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(Blob1, Blob2, NumEqualBytes,
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
