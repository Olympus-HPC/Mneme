#define NOMINMAX

#include <algorithm> //min  max
#include <vector>
#include <math.h>
#include <cuda.h>

#include "indices.hpp"
#include "params.hpp"

//2DDCT - has to be consistent with dct8x8.cu
#define KER2_BLOCK_WIDTH          128

//Exception handling
#include <sstream>
//Debug
#include <fstream>
#include <iostream>

//Extern kernels

extern "C" void run_block_matching(
  const  uchar* __restrict image,
  ushort* stacks,
  uint* num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const uint2 start_point,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

extern "C" void run_get_block(
  const uint2 start_point,
  const uchar* __restrict image,
  const ushort* __restrict stacks,
  const uint* __restrict num_patches_in_stack,
  float* patch_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_DCT2D8x8(
  float *d_transformed_stacks,
  const float *d_gathered_stacks,
  const uint size,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_hard_treshold_block(
  const uint2 start_point,
  float* patch_stack,
  float* w_P,
  const uint* __restrict num_patches_in_stack,
  const uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

extern "C" void run_IDCT2D8x8(
  float *d_gathered_stacks,
  const float *d_transformed_stacks,
  const uint size,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_aggregate_block(
  const uint2 start_point,
  const float* __restrict patch_stack,  
  const float* __restrict w_P,
  const ushort* __restrict stacks,
  const float* __restrict kaiser_window,
  float* numerator,
  float* denominator,
  const uint* __restrict num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_aggregate_final(
  const float* __restrict numerator,
  const float* __restrict denominator,
  const uint2 image_dim,
  uchar* denoised_noisy_image,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_wiener_filtering(
  const uint2 start_point,
  float* patch_stack,
  const float* __restrict patch_stack_basic,
  float*  w_P,
  const uint* __restrict num_patches_in_stack,
  uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

// error handling
#define cuda_error_check(ans) { display_cuda_error((ans),__FILE__, __LINE__); }
void display_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << "): " << cudaGetErrorString(code);
    std::string file_and_line;
    ss >> file_and_line;
  }
}

