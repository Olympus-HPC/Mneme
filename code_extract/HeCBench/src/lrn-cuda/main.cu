#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "kernels.h"

using namespace std::chrono;

void Forward(int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 6;
  int64_t C = 150;
  int64_t D = 100;
  int64_t H = 160;
  int64_t W = 160;
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) { 
    src[i] = rand() / (float)RAND_MAX;
  }

  float *src_mem;
  size_t bytes_to_copy_s = wk_size * sizeof(float);
  cudaMalloc((void**)&src_mem, bytes_to_copy_s);
  cudaMemcpy(src_mem, src.data(), bytes_to_copy_s, cudaMemcpyHostToDevice);

  size_t bytes_to_copy_d = wk_size * sizeof(float);
  float *dst_mem;
  cudaMalloc((void**)&dst_mem, bytes_to_copy_d);
  cudaMemcpy(dst_mem, dst.data(), bytes_to_copy_d, cudaMemcpyHostToDevice);

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    cudaDeviceSynchronize();
    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      lrn_fwd_kernel<<<wg_cnt, wg_size>>>(
        src_mem, dst_mem, N, C, D, H, W, stride_mb, ndims, wk_size, size, alpha, beta, k);
    }

    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_fwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (2 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }

  cudaMemcpy(dst.data(), dst_mem, bytes_to_copy_d, cudaMemcpyDeviceToHost);
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) { 
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);

  cudaFree(src_mem);
  cudaFree(dst_mem);
}


void Backward(int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 5;
  int64_t C = 150;
  int64_t D = 100;
  int64_t H = 160;
  int64_t W = 160;
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);
  std::vector<float> diff_src(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) { 
    diff_src[i] = src[i] = rand() / (float)RAND_MAX;
  }

  float *src_mem;
  size_t bytes_to_copy_s = wk_size * sizeof(float);
  cudaMalloc((void**)&src_mem, bytes_to_copy_s);
  cudaMemcpy(src_mem, src.data(), bytes_to_copy_s, cudaMemcpyHostToDevice);

  float *diff_src_mem;
  size_t bytes_to_copy_diff = wk_size * sizeof(float);
  cudaMalloc((void**)&diff_src_mem, bytes_to_copy_diff);
  cudaMemcpy(diff_src_mem, diff_src.data(), bytes_to_copy_diff, cudaMemcpyHostToDevice);

  float *dst_mem;
  size_t bytes_to_copy_d = wk_size * sizeof(float);
  cudaMalloc((void**)&dst_mem, bytes_to_copy_d);
  cudaMemcpy(dst_mem, src.data(), bytes_to_copy_d, cudaMemcpyHostToDevice);

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    cudaDeviceSynchronize();
    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      lrn_bwd_kernel<<<wg_cnt, wg_size>>>(
        src_mem, dst_mem, diff_src_mem, N, C, D, H, W, stride_mb, ndims, wk_size, size, alpha, beta, k);
    }

    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_bwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (3 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }


  cudaMemcpy(dst.data(), dst_mem, bytes_to_copy_d, cudaMemcpyDeviceToHost);
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) { 
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);
  cudaFree(src_mem);
  cudaFree(diff_src_mem);
  cudaFree(dst_mem);
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Forward(repeat);
  Backward(repeat);

  return 0;
}
