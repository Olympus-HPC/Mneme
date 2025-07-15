#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "gdt/math/vec.h"
#include "utils.h"

using vec3i = gdt::vec3i;

template<int WIN_SIZE>
__global__ void 
compute_ssim(const uint32_t dimx, const uint32_t dimy, const uint32_t dimz,
             float* __restrict__ _fx, float* __restrict__ _fy, vec3i gdims,              
             float data_range, float cov_norm, float K1, float K2,
             float* __restrict__ out)
{
  const int32_t x = blockIdx.x * blockDim.x + threadIdx.x; if (x >= dimx) return;
  const int32_t y = blockIdx.y * blockDim.y + threadIdx.y; if (y >= dimy) return;
  const int32_t z = blockIdx.z * blockDim.z + threadIdx.z; if (z >= dimz) return;

  float ux = 0.f, uy = 0.f, uxx = 0.f, uyy = 0.f, uxy = 0.f;

  for (int kz = 0; kz < WIN_SIZE; ++kz) {
  for (int ky = 0; ky < WIN_SIZE; ++ky) {
  for (int kx = 0; kx < WIN_SIZE; ++kx) {

    const vec3i g = vec3i(x + kx, y + ky, z + kz);
    const uint32_t gidx = g.x + g.y * gdims.x + g.z * gdims.x * gdims.y;
    const float fx = _fx[gidx];
    const float fy = _fy[gidx];

    ux  += fx;
    uy  += fy;    
    uxx += fx * fx;
    uyy += fy * fy;
    uxy += fx * fy;

  } } }
    
  const float w = 1.f / (WIN_SIZE*WIN_SIZE*WIN_SIZE); // uniform filter
  ux  *= w;
  uy  *= w;
  uxx *= w;
  uyy *= w;
  uxy *= w;

  const float vx = cov_norm * (uxx - ux * ux);
  const float vy = cov_norm * (uyy - uy * uy);
  const float vxy = cov_norm * (uxy - ux * uy);

  const float R = data_range;
  const float C1 = (K1 * R) * (K1 * R);
  const float C2 = (K2 * R) * (K2 * R);

  const float A1 = 2.f * ux * uy + C1;
  const float A2 = 2.f * vxy + C2;
  const float B1 = ux * ux + uy * uy + C1;
  const float B2 = vx + vy + C2;
  const float D = B1 * B2;
  const float S = (A1 * A2) / D;

  out[x + y * dimx + z * dimx * dimy] = S;
}

int main() {
  const vec3i dims(4096, 1024, 1024); // inputs are volumetric data
  constexpr float K1 = 0.01f; // paramemters of SSIM
  constexpr float K2 = 0.03f;
  const float data_range = 1.f; // range of the image

  constexpr bool use_sample_covariance = true;

  constexpr int win_size = 7; // kernel size (backwards compatibility)
  constexpr int crop = win_size >> 1;
  constexpr int NP = win_size * win_size * win_size;
  // filter has already normalized by NP
  constexpr float cov_norm = use_sample_covariance 
                             ? (float)NP / (NP - 1) // sample covariance
                             : 1.f; // population covariance to match Wang et. al. 2004

  const vec3i batch = min(vec3i(4096,16,16),dims);

  const vec3i batch_grid = batch + win_size - 1;
  const auto batch_grid_count = util::next_multiple<size_t>(batch_grid.long_product(), 256);

  const size_t input_size = sizeof(float) * batch_grid_count; 
  const size_t output_size = input_size * 3;

  float *h_grid_inference = (float*) malloc (input_size);
  float *h_grid_reference = (float*) malloc (input_size);

  srand(123);
  for (size_t i = 0; i < batch_grid_count; i++) {
    h_grid_reference[i] = rand() / (float)RAND_MAX;
    h_grid_inference[i] = h_grid_reference[i] * 0.75f;
  }

  float *grid_output;
  CUDA_CHECK(cudaMalloc((void**)&grid_output, output_size));

  float *grid_inference;
  CUDA_CHECK(cudaMalloc((void**)&grid_inference, input_size));

  float *grid_reference;
  CUDA_CHECK(cudaMalloc((void**)&grid_reference, input_size));
  
  CUDA_CHECK(cudaMemcpy(grid_reference, h_grid_reference, input_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(grid_inference, h_grid_inference, input_size, cudaMemcpyHostToDevice));

  float ssim_sum = 0.f;
  long etime = 0;
  for (int z = crop; z < dims.z - crop; z += batch.z) {
  for (int y = crop; y < dims.y - crop; y += batch.y) {
  for (int x = crop; x < dims.x - crop; x += batch.x) {
    const vec3i block_offset = vec3i(x,y,z);
    const vec3i block = min(batch, dims - crop - block_offset);
    const auto block_count = block.long_product();
    if (block_count == 0) continue;

    // compute grid values
    const vec3i block_grid = block + win_size - 1;

    auto start = std::chrono::steady_clock::now();

    // calculate SSIM between reference and inference
    util::trilinear_kernel(compute_ssim<win_size>, 0, 0, block.x, block.y, block.z, 
                     grid_reference, grid_inference, block_grid,
                     data_range, cov_norm, K1, K2, grid_output);

    // compute total ssim
    const auto begin = thrust::device_ptr<float>(grid_output); // wrap raw pointer with a device_ptr
    ssim_sum += thrust::reduce(begin, begin + block_count, (float) 0, thrust::plus<float>());

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    etime += time;
  } } }

  CUDA_CHECK(cudaFree(grid_reference));
  CUDA_CHECK(cudaFree(grid_inference));
  CUDA_CHECK(cudaFree(grid_output));
  free(h_grid_reference);
  free(h_grid_inference);

  printf("Total kernel execution time (s): %lf\n", etime * 1e-9);
  printf("Structural Similarity Index Measure: %f\n",
          ssim_sum / (dims - win_size + 1).long_product());

  return 0;
}
