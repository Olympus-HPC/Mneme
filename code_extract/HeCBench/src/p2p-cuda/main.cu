/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

__global__ void SimpleKernel(const float *src, float *dst)
{
  // Just a dummy kernel, doing enough for us to verify that everything
  // worked
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = src[idx] * 2.0f;
}

inline bool IsAppBuiltAs64()
{
  return sizeof(void*) == 8;
}

int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);
  const int repeat = atoi(argv[1]);

  if (!IsAppBuiltAs64())
  {
    printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
    exit(0);
  }

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  int gpu_n;
  cudaGetDeviceCount(&gpu_n);
  printf("There are %d GPUs\n", gpu_n);

  if (gpu_n < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Waiving test.\n");
    exit(0);
  }

  // Query device properties
  cudaDeviceProp prop[64];

  for (int i=0; i < gpu_n; i++)
  {
    cudaGetDeviceProperties(&prop[i], i);
  }
  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  int can_access_peer;
  int p2pCapableGPUs[2] = {-1, -1}; // We take only 1 pair of P2P capable GPUs

  // Show all the combinations of supported P2P GPUs
  for (int i = 0; i < gpu_n; i++)
  {
    for (int j = 0; j < gpu_n; j++)
    {
      if (i == j)
      {
        continue;
      }
      cudaDeviceCanAccessPeer(&can_access_peer, i, j);
      printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name, i,
             prop[j].name, j, can_access_peer ? "Yes" : "No");
      if (can_access_peer && p2pCapableGPUs[0] == -1)
      {
        p2pCapableGPUs[0] = i;
        p2pCapableGPUs[1] = j;
      }
    }
  }

  if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Peer to Peer access is not available amongst GPUs in the system, waiving test.\n");
    exit(0);
  }

  // Use first pair of p2p capable GPUs detected.
  int gpuid[2]; // we want to find the first two GPU's that can support P2P
  gpuid[0] = p2pCapableGPUs[0];
  gpuid[1] = p2pCapableGPUs[1];

  // Enable peer access
  printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0], gpuid[1]);
  cudaSetDevice(gpuid[0]);
  cudaDeviceEnablePeerAccess(gpuid[1], 0);

  cudaSetDevice(gpuid[1]);
  cudaDeviceEnablePeerAccess(gpuid[0], 0);

  // Allocate buffers
  const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n",
         int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);

  // GPU0
  cudaSetDevice(gpuid[0]);
  float *g0;
  cudaMalloc(&g0, buf_size);

  float *h0;
  cudaMallocHost(&h0, buf_size); // Automatically portable with UVA

  // GPU1
  cudaSetDevice(gpuid[1]);
  float *g1;
  cudaMalloc(&g1, buf_size);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i=0; i<repeat; i++)
  {
    // With UVA we don't need to specify source and target devices, the
    // runtime figures this out by itself from the pointers
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault);
    }
    else
    {
      cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault);
    }
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time_memcpy = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("Peer-to-peer copy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
         1.0f / time_memcpy * (repeat * buf_size));

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

  const int buf_len = buf_size / sizeof(float);
  for (int i=0; i<buf_len; i++)
  {
    h0[i] = float(i % 4096);
  }

  cudaSetDevice(gpuid[0]);
  cudaMemcpy(g0, h0, buf_size, cudaMemcpyDefault);

  // Kernel launch configuration
  const dim3 threads(256);
  const dim3 blocks(buf_len / 256);

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
  // output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
         gpuid[1], gpuid[0], gpuid[1]);

  cudaSetDevice(gpuid[1]);
  SimpleKernel<<<blocks, threads>>>(g0, g1);

  cudaDeviceSynchronize();

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
  // output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
         gpuid[0], gpuid[1], gpuid[0]);

  cudaSetDevice(gpuid[0]);
  SimpleKernel<<<blocks, threads>>>(g1, g0);

  cudaDeviceSynchronize();

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
  cudaMemcpy(h0, g0, buf_size, cudaMemcpyDefault);

  int error_count = 0;

  for (int i=0; i<buf_len; i++)
  {
    // Re-generate input data and apply 2x '* 2.0f' computation of both
    // kernel runs
    if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i%4096)*2.0f*2.0f));

      if (error_count++ > 10)
      {
        break;
      }
    }
  }

  // Disable peer access (also unregisters memory for non-UVA cases)
  printf("Disabling peer access...\n");
  cudaSetDevice(gpuid[0]);
  cudaDeviceDisablePeerAccess(gpuid[1]);

  cudaSetDevice(gpuid[1]);
  cudaDeviceDisablePeerAccess(gpuid[0]);

  // Cleanup and shutdown
  printf("Shutting down...\n");
  cudaSetDevice(gpuid[0]);
  cudaFree(g0);
  cudaFreeHost(h0);

  cudaSetDevice(gpuid[1]);
  cudaFree(g1);

  if (error_count != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }
  else
  {
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
  }
}
