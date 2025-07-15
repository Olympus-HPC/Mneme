#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "kernels.h"

#define nthreads 256

int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  int size = width * height;
  size_t size_output_bytes = size * sizeof(uint);
  size_t size_image_bytes = size * sizeof(float3);

  std::mt19937 gen(19937);
  // reduce the upper bound can increase the kernel execution time of eliminate_crosses
  std::uniform_real_distribution<float> dis(0.f, 0.4f); 

  float3 *h_img = (float3*) malloc(size_image_bytes);

  uint *h_out = (uint*) malloc(size_output_bytes);

  float3 *d_img;
  cudaMalloc((void**)&d_img, size_image_bytes);

  uint *d_tmp, *d_out;
  cudaMalloc((void**)&d_tmp, size_output_bytes);
  cudaMalloc((void**)&d_out, size_output_bytes);

  // assume that size is a multiple of nthreads
  dim3 grids (size / nthreads); 
  dim3 blocks (nthreads);

  float sum = 0;
  float total_time = 0;

  for (int n = 0; n < repeat; n++) {

    for (int i = 0; i < size; i++) {
      h_img[i].x = dis(gen);
      h_img[i].y = dis(gen);
      h_img[i].z = dis(gen);
    }

    cudaMemcpy(d_img, h_img, size_image_bytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    check_connect<<<grids, blocks>>>(d_img, d_tmp, width, height);
    eliminate_crosses<<<grids, blocks>>>(d_tmp, d_out, width, height);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    total_time += time.count();

    cudaMemcpy(h_out, d_out, size_output_bytes, cudaMemcpyDeviceToHost);

    float lsum = 0;
    for (int i = 0; i < size; i++)
      lsum += (h_out[i] & 0xff) / 256.f + 
             ((h_out[i] >> 8) & 0xff) / 256.f + 
             ((h_out[i] >> 16) & 0xff) / 256.f + 
             ((h_out[i] >> 24) & 0xff) / 256.f;

    sum += lsum / size;
  }

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n",
         width, height, sum);
  printf("Average kernel time over %d iterations: %f (s)\n",
         repeat, total_time / repeat);

  cudaFree(d_out);
  cudaFree(d_img);
  cudaFree(d_tmp);
  free(h_out);
  free(h_img);

  return 0;
}
