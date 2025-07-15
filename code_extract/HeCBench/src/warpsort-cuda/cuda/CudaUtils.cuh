// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda.h>

namespace facebook { namespace cuda {

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ constexpr T ceil(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes floor(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ constexpr T floor(T a, T b) {
  return (a - b + 1) / b;
}

/**
   Returns the current thread's warp ID
*/
__device__ __forceinline__ int getWarpId() {
  return threadIdx.x / warpSize;
}

/**
   Returns the number of threads in the current block (linearized).
*/
__device__ __forceinline__ int getThreadsInBlock() {
  return blockDim.x;
}

/**
   Returns the number of warps in the current block (linearized,
   rounded to whole warps).
*/
__device__ __forceinline__ int getWarpsInBlock() {
  return ceil(getThreadsInBlock(), WARP_SIZE);
}


/**
   Return the current thread's lane in the warp
*/
__device__ __forceinline__ int getLaneId() {
  int laneId = threadIdx.x % WARP_SIZE;
  return laneId;
}

/**
   Extract a single bit at `pos` from `val`
*/

__device__ __forceinline__ int getBit(int val, int pos) {
  return (val >> pos) & 0x1;
}

/**
   Returns the index of the most significant 1 bit in `val`.
*/

__device__ __forceinline__ constexpr int getMSB(int val) {
  return
    ((val >= 1024 && val < 2048) ? 10 :
     ((val >= 512) ? 9 :
      ((val >= 256) ? 8 :
       ((val >= 128) ? 7 :
        ((val >= 64) ? 6 :
         ((val >= 32) ? 5 :
          ((val >= 16) ? 4 :
           ((val >= 8) ? 3 :
            ((val >= 4) ? 2 :
             ((val >= 2) ? 1 :
              ((val == 1) ? 0 : -1)))))))))));
}

} }  // namespace
