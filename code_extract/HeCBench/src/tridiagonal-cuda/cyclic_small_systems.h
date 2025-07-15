/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Tridiagonal solvers.
 * Host code for cyclic reduction (CR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _CYCLIC_SMALL_SYSTEMS_
#define _CYCLIC_SMALL_SYSTEMS_

#include <cuda.h>
#include "tridiagonal.h"
#include "cyclic_kernels.cu"

const char *cyclicKernelNames[] = { 
  "cyclic_small_systems_kernel",    // original version
  "cyclic_branch_free_kernel",      // optimized branch-free version
};  

double cyclic_small_systems(float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, int id = 0)
{
  shrLog(" %s\n", cyclicKernelNames[id]);

  const unsigned int mem_size = num_systems * system_size * sizeof(float);
  float* a_d;
  cudaMalloc((void**)&a_d, mem_size);
  cudaMemcpy(a_d, a, mem_size, cudaMemcpyHostToDevice); 

  float* b_d;
  cudaMalloc((void**)&b_d, mem_size);
  cudaMemcpy(b_d, b, mem_size, cudaMemcpyHostToDevice); 

  float* c_d;
  cudaMalloc((void**)&c_d, mem_size); 
  cudaMemcpy(c_d, c, mem_size, cudaMemcpyHostToDevice);

  float* d_d;
  cudaMalloc((void**)&d_d, mem_size); 
  cudaMemcpy(d_d, d, mem_size, cudaMemcpyHostToDevice);

  float* x_d;
  cudaMalloc((void**)&x_d, mem_size); 

  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;
  int iterations = log2(system_size/2);

  // set execution parameters
  szLocalWorkSize = system_size / 2;
  szGlobalWorkSize = num_systems; 

  dim3 gws (szGlobalWorkSize);
  dim3 lws (szLocalWorkSize);

  // warm up
  if (id == 0)
    cyclic_small_systems_kernel<<<gws, lws, (system_size+1)*5*sizeof(float)>>>(
        a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);
  else
    cyclic_branch_free_kernel<<<gws, lws, (system_size+1)*5*sizeof(float)>>>( 
        a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);

  cudaDeviceSynchronize();

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (id == 0)
      cyclic_small_systems_kernel<<<gws, lws, (system_size+1)*5*sizeof(float)>>>(
          a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);
    else
      cyclic_branch_free_kernel<<<gws, lws, (system_size+1)*5*sizeof(float)>>>( 
          a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);
  }
  cudaDeviceSynchronize();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  // write-back to the array x
  cudaMemcpy(x, x_d, mem_size, cudaMemcpyDeviceToHost); 

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  cudaFree(x_d);
  return time;
}

#endif
