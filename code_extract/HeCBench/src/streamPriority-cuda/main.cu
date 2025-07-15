/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdio>
#include <iostream>
#include <cuda.h>

#define EACH_SIZE 256 * 1024

// # threadblocks
#define TBLOCKS 256
#define THREADS 256

#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
  if (cudaSuccess != err)
  {   
    std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file "
              << file  << ", line " << line << std::endl;
  }
}
#endif

// throw error on equality
#define ERR_EQ(X, Y)                                                           \
  do {                                                                         \
    if ((X) == (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

// throw error on difference
#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

// copy from source -> destination arrays
__global__ void memcpy_kernel(int *dst, int *src, size_t n, bool wait) {
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t i = id; i < n / sizeof(int); i += num) {
    int v = src[i];
    if (wait) {
      while (v--) 
        dst[i] = v;
    }
    dst[i] = src[i];
  }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (size_t i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

long eval (cudaStream_t &s1, cudaStream_t &s2) {
  size_t size = 1UL << 29;

  // initialise host data
  int *h_src_low;
  int *h_src_hi;
  ERR_EQ(h_src_low = (int *)malloc(size), NULL);
  ERR_EQ(h_src_hi = (int *)malloc(size), NULL);
  mem_init(h_src_low, size);
  mem_init(h_src_hi, size);

  // initialise device data
  int *h_dst_low;
  int *h_dst_hi;
  ERR_EQ(h_dst_low = (int *)malloc(size), NULL);
  ERR_EQ(h_dst_hi = (int *)malloc(size), NULL);
  memset(h_dst_low, 0, size);
  memset(h_dst_hi, 0, size);

  // copy source data -> device
  int *d_src_low;
  int *d_src_hi;
  checkCudaErrors(cudaMalloc(&d_src_low, size));
  checkCudaErrors(cudaMalloc(&d_src_hi, size));
  checkCudaErrors(cudaMemcpy(d_src_low, h_src_low, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_src_hi, h_src_hi, size, cudaMemcpyHostToDevice));

  // allocate memory for memcopy destination
  int *d_dst_low;
  int *d_dst_hi;
  checkCudaErrors(cudaMalloc(&d_dst_low, size));
  checkCudaErrors(cudaMalloc(&d_dst_hi, size));

  // warmup
  for (size_t i = 0; i < size; i += EACH_SIZE) {
    size_t j = i / sizeof(int);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, s1>>>(d_dst_low + j, d_src_low + j, EACH_SIZE, true);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, s2>>>(d_dst_hi + j, d_src_hi + j, EACH_SIZE, false);
  }
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < size; i += EACH_SIZE) {
    size_t j = i / sizeof(int);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, s1>>>(d_dst_low + j, d_src_low + j, EACH_SIZE, true);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, s2>>>(d_dst_hi + j, d_src_hi + j, EACH_SIZE, false);
  }
  cudaStreamSynchronize(s2);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  checkCudaErrors(cudaMemcpy(h_dst_low, d_dst_low, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dst_hi, d_dst_hi, size, cudaMemcpyDeviceToHost));

  // check results of kernels
  ERR_NE(memcmp(h_dst_low, h_src_low, size), 0);
  ERR_NE(memcmp(h_dst_hi, h_src_hi, size), 0);

  checkCudaErrors(cudaFree(d_src_low));
  checkCudaErrors(cudaFree(d_src_hi));
  checkCudaErrors(cudaFree(d_dst_low));
  checkCudaErrors(cudaFree(d_dst_hi));
  free(h_src_low);
  free(h_src_hi);
  free(h_dst_low);
  free(h_dst_hi);

  return time;
}

int main(int argc, char **argv) {

  printf("Starting [%s]...\n", argv[0]);

  // get the range of priorities available
  // [ greatest_priority, lowest_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));
  printf("Stream priority range: low: %d to high: %d\n", priority_low, priority_hi);

  // create streams with highest and lowest available priorities
  cudaStream_t s1, s2;
  checkCudaErrors(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, priority_low));
  checkCudaErrors(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, priority_hi));

  auto time = eval(s1, s2);
  printf("Elapsed time of kernel launched to high priority stream: %.3lf ms\n", time * 1e-6);

  cudaStream_t s3, s4;
  checkCudaErrors(cudaStreamCreate(&s3));
  checkCudaErrors(cudaStreamCreate(&s4));

  time = eval(s3, s4);
  printf("Elapsed time of kernel launched to no-priority stream: %.3lf ms\n", time * 1e-6);

  checkCudaErrors(cudaStreamDestroy(s1));
  checkCudaErrors(cudaStreamDestroy(s2));
  checkCudaErrors(cudaStreamDestroy(s3));
  checkCudaErrors(cudaStreamDestroy(s4));

  return 0;
}
