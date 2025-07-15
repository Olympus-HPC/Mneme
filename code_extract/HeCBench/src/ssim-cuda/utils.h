//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif 

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t error = call;                                                                  \
    if (error != cudaSuccess) {                                                                \
      const char* msg = cudaGetErrorString(error);                                             \
      fprintf(stderr, "CUDA error (%s: line %d): %s\n", __FILE__, __LINE__, msg);              \
      throw std::runtime_error(std::string("CUDA error: " #call " failed with error ") + msg); \
    }                                                                                          \
  } while (0)


#ifdef __NVCC__
#define CUDA_UTIL_BOTH_INLINE __forceinline__ __device__ __host__
#else
#define CUDA_UTIL_BOTH_INLINE 
#endif

namespace util {

  template <typename T>
    CUDA_UTIL_BOTH_INLINE T div_round_up(T val, T divisor) {
      return (val + divisor - 1) / divisor;
    }

  template <typename T>
    CUDA_UTIL_BOTH_INLINE T next_multiple(T val, T divisor) {
      return div_round_up(val, divisor) * divisor;
    }

  constexpr uint32_t n_threads_trilinear = 8;

#ifdef __NVCC__

  //. ======================================================================== //
  // trilinear version 
  //. ======================================================================== //

  template<typename T>
    constexpr uint32_t
    n_blocks_trilinear(T n_elements)
    {
      return ((uint32_t)n_elements + n_threads_trilinear - 1) / n_threads_trilinear;
    }

  template<typename K, typename T, typename... Types>
    inline void
    trilinear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T width, T height, T depth, Types... args)
    {
      if (width <= 0 || height <= 0 || depth <= 0) {
        return;
      }
      dim3 block_size(n_threads_trilinear, n_threads_trilinear, n_threads_trilinear);
      dim3 grid_size(n_blocks_trilinear(width), n_blocks_trilinear(height), n_blocks_trilinear(depth));
      kernel<<<grid_size, block_size, shmem_size, stream>>>((uint32_t)width, (uint32_t)height, (uint32_t)depth, args...);
    }

  template<typename K, typename... Types>
    inline void
    trilinear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, int3 dims, Types... args)
    {
      if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        return;
      }
      dim3 block_size(n_threads_trilinear, n_threads_trilinear, n_threads_trilinear);
      dim3 grid_size(n_blocks_trilinear(dims.x), n_blocks_trilinear(dims.y), n_blocks_trilinear(dims.z));
      kernel<<<grid_size, block_size, shmem_size, stream>>>(dims, args...);
    }

#endif
}
