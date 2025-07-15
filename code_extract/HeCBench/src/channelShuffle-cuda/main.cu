#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

#define NUM_THREADS 256
#define GridDimMaxY 65536

template <typename T, bool kNFirst>
__global__ void ChannelShuffleNCHWKernel(
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  const int C = G * K;
  const int n = kNFirst ? blockIdx.x : blockIdx.y;
  const int s = kNFirst ? blockIdx.y : blockIdx.x;
  const int g = blockIdx.z % G;
  const int k = blockIdx.z / G;
  const int offset = s * NUM_THREADS + threadIdx.x;
  if (offset < HxW) {
    Y[(n * C + blockIdx.z) * HxW + offset] =
        __ldg(X + (n * C + g * K + k) * HxW + offset);
  }
}

template <typename T, int kSharedSize>
__global__ void
ChannelShuffleNHWCKernel(const int G, const int K, const T* X, T* Y)
{
  __shared__ T sdata[kSharedSize];
  const int C = G * K;
  const int offset = blockIdx.x * C;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    sdata[i] = __ldg(X + offset + i);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    const int g = i % G;
    const int k = i / G;
    Y[offset + i] = sdata[g * K + k];
  }
}

template <typename T>
bool ChannelShuffleNCHW (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  if (N <= GridDimMaxY) {
    const dim3 dim_grid(S, N, C);
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNCHWKernel<float, false>
        <<<dim_grid, NUM_THREADS>>>(G, K, HxW, X, Y);
  } else {
    const dim3 dim_grid(N, S, C);
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNCHWKernel<float, true>
        <<<dim_grid, NUM_THREADS>>>(G, K, HxW, X, Y);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int outer_size = N * HxW;

  auto start = std::chrono::steady_clock::now();

  if (C <= 32) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 32>
        <<<outer_size, NUM_THREADS>>>(G, K, X, Y);
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 128>
        <<<outer_size, NUM_THREADS>>>(G, K, X, Y);
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 512>
        <<<outer_size, NUM_THREADS>>>(G, K, X, Y);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int G = atoi(argv[1]);
  const int W = atoi(argv[2]);
  const int H = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  long time;
  float *h_X, *h_Y, *h_Y_ref;
  float *d_X, *d_Y;
  int error;

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H; // assume no integer overflow
      size_t data_size_bytes = numel * sizeof(float);

      auto errorX = cudaMalloc((void**)&d_X, data_size_bytes);
      auto errorY = cudaMalloc((void**)&d_Y, data_size_bytes);
      if (errorX != cudaSuccess || errorY != cudaSuccess) {
        if (errorX == cudaSuccess) cudaFree(d_X);
        if (errorY == cudaSuccess) cudaFree(d_Y);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      h_X = (float*) malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float) i / numel;

      h_Y = (float*) malloc(data_size_bytes);
      h_Y_ref = (float*) malloc(data_size_bytes);

      cudaMemcpy(d_X, h_X, data_size_bytes, cudaMemcpyHostToDevice);

      ChannelShuffleNHWC (d_X, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNHWC_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      cudaMemcpy(h_Y, d_Y, data_size_bytes, cudaMemcpyDeviceToHost);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NHWC) check\n");
      else
        printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

      ChannelShuffleNCHW (d_X, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNCHW_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      cudaMemcpy(h_Y, d_Y, data_size_bytes, cudaMemcpyDeviceToHost);
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NCHW) check\n");
      else
        printf("Average time of channel shuffle (NCHW): %f (ms)\n", (time * 1e-6f) / repeat);

      cudaFree(d_X);
      cudaFree(d_Y);
      free(h_X);
      free(h_Y);
      free(h_Y_ref);
    }
  }

  end: return 0;
}
