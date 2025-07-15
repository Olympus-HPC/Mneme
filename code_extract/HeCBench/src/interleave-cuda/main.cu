/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 */
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

#define NUM_ELEMENTS 4096
#define COUNT 4096       // accumulation count

// an interleaved type
typedef struct
{
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
  unsigned int s3;
  unsigned int s4;
  unsigned int s5;
  unsigned int s6;
  unsigned int s7;
  unsigned int s8;
  unsigned int s9;
  unsigned int sa;
  unsigned int sb;
  unsigned int sc;
  unsigned int sd;
  unsigned int se;
  unsigned int sf;
} INTERLEAVED_T;

// Define an array type based on the interleaved structure
typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

// Alternative - structure of arrays
typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct
{
  ARRAY_MEMBER_T s0;
  ARRAY_MEMBER_T s1;
  ARRAY_MEMBER_T s2;
  ARRAY_MEMBER_T s3;
  ARRAY_MEMBER_T s4;
  ARRAY_MEMBER_T s5;
  ARRAY_MEMBER_T s6;
  ARRAY_MEMBER_T s7;
  ARRAY_MEMBER_T s8;
  ARRAY_MEMBER_T s9;
  ARRAY_MEMBER_T sa;
  ARRAY_MEMBER_T sb;
  ARRAY_MEMBER_T sc;
  ARRAY_MEMBER_T sd;
  ARRAY_MEMBER_T se;
  ARRAY_MEMBER_T sf;
} NON_INTERLEAVED_T;

// data initialization and verification
#include "util.cpp"

__global__ void add_kernel_interleaved(
    INTERLEAVED_T * const dest_ptr,
    const INTERLEAVED_T * const src_ptr,
    const unsigned int num_elements)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_elements)
  {
    for (unsigned int i=0; i<COUNT; i++)
    {
      dest_ptr[tid].s0 += src_ptr[tid].s0;
      dest_ptr[tid].s1 += src_ptr[tid].s1;
      dest_ptr[tid].s2 += src_ptr[tid].s2;
      dest_ptr[tid].s3 += src_ptr[tid].s3;
      dest_ptr[tid].s4 += src_ptr[tid].s4;
      dest_ptr[tid].s5 += src_ptr[tid].s5;
      dest_ptr[tid].s6 += src_ptr[tid].s6;
      dest_ptr[tid].s7 += src_ptr[tid].s7;
      dest_ptr[tid].s8 += src_ptr[tid].s8;
      dest_ptr[tid].s9 += src_ptr[tid].s9;
      dest_ptr[tid].sa += src_ptr[tid].sa;
      dest_ptr[tid].sb += src_ptr[tid].sb;
      dest_ptr[tid].sc += src_ptr[tid].sc;
      dest_ptr[tid].sd += src_ptr[tid].sd;
      dest_ptr[tid].se += src_ptr[tid].se;
      dest_ptr[tid].sf += src_ptr[tid].sf;
    }
  }
}

__global__ void add_kernel_non_interleaved(
    NON_INTERLEAVED_T * const dest_ptr,
    const NON_INTERLEAVED_T * const src_ptr,
    const unsigned int num_elements)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_elements)
  {
    for (unsigned int i=0; i<COUNT; i++)
    {
      dest_ptr->s0[tid] += src_ptr->s0[tid];
      dest_ptr->s1[tid] += src_ptr->s1[tid];
      dest_ptr->s2[tid] += src_ptr->s2[tid];
      dest_ptr->s3[tid] += src_ptr->s3[tid];
      dest_ptr->s4[tid] += src_ptr->s4[tid];
      dest_ptr->s5[tid] += src_ptr->s5[tid];
      dest_ptr->s6[tid] += src_ptr->s6[tid];
      dest_ptr->s7[tid] += src_ptr->s7[tid];
      dest_ptr->s8[tid] += src_ptr->s8[tid];
      dest_ptr->s9[tid] += src_ptr->s9[tid];
      dest_ptr->sa[tid] += src_ptr->sa[tid];
      dest_ptr->sb[tid] += src_ptr->sb[tid];
      dest_ptr->sc[tid] += src_ptr->sc[tid];
      dest_ptr->sd[tid] += src_ptr->sd[tid];
      dest_ptr->se[tid] += src_ptr->se[tid];
      dest_ptr->sf[tid] += src_ptr->sf[tid];
    }
  }
}

void add_test_interleaved(
    INTERLEAVED_T * const h_dst,
    const INTERLEAVED_T * const h_src,
    const int repeat,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int num_threads = 256;
  const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;

  // Allocate memory on the device
  const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
  INTERLEAVED_T * d_dst;
  INTERLEAVED_T * d_src;
  cudaMalloc((void **) &d_src, num_bytes);
  cudaMalloc((void **) &d_dst, num_bytes);
  cudaMemcpy(d_src, h_src, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst, h_dst, num_bytes, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
    add_kernel_interleaved<<<num_blocks, num_threads>>>(d_dst, d_src, num_elements);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (interleaved) execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(h_dst, d_dst, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_src);
  cudaFree(d_dst);
}

void add_test_non_interleaved(
    NON_INTERLEAVED_T * const h_dst,
    const NON_INTERLEAVED_T * const h_src,
    const int repeat,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int num_threads = 256;
  const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;
  // Allocate memory on the device
  const size_t num_bytes = sizeof(NON_INTERLEAVED_T);
  //printf("%lu\n", num_bytes);
  NON_INTERLEAVED_T * d_dst;
  NON_INTERLEAVED_T * d_src;
  cudaMalloc((void **) &d_src, num_bytes);
  cudaMalloc((void **) &d_dst, num_bytes);
  cudaMemcpy(d_src, h_src, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst, h_dst, num_bytes, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
    add_kernel_non_interleaved<<<num_blocks, num_threads>>>(d_dst, d_src, num_elements);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (non-interleaved) execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(h_dst, d_dst, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_src);
  cudaFree(d_dst);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst; 
  INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst; 
  initialize (interleaved_src, interleaved_dst, 
              non_interleaved_src, non_interleaved_dst, NUM_ELEMENTS);
  add_test_non_interleaved(&non_interleaved_dst, &non_interleaved_src,
                           repeat, NUM_ELEMENTS);
  add_test_interleaved(interleaved_dst, interleaved_src, repeat, NUM_ELEMENTS);
  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);
  return 0;
}
