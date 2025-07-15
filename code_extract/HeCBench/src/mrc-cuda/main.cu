#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "reference.h"

__global__
void MRCGradient (
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}

__global__
void MRCGradient2(
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = -y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.f ? 0.f : -y * o;
    dX2[i] = dist < 0.f ? 0.f : y * o;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t size_bytes = length * sizeof(float);

  float *h_X1  = (float*) malloc (size_bytes);
  float *h_X2  = (float*) malloc (size_bytes);
  float *h_O   = (float*) malloc (size_bytes);
    int *h_Y   = (  int*) malloc (size_bytes);
  float *h_dX1 = (float*) malloc (size_bytes);
  float *h_dX2 = (float*) malloc (size_bytes);
  float *r_dX1 = (float*) malloc (size_bytes);
  float *r_dX2 = (float*) malloc (size_bytes);

  const float m = 0.01;  // margin

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < length; i++) {
    h_X1[i] = distr(g);
    h_X2[i] = distr(g);
    h_O[i] = distr(g);
    h_Y[i] = (distr(g) < 0) ? -1 : 1;
  }

  float *d_X1, *d_X2, *d_O, *d_dX1, *d_dX2;
  int *d_Y;
  cudaMalloc((void**)&d_X1, size_bytes);
  cudaMemcpy(d_X1, h_X1, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_X2, size_bytes);
  cudaMemcpy(d_X2, h_X2, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_O, size_bytes);
  cudaMemcpy(d_O, h_O, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_Y, size_bytes);
  cudaMemcpy(d_Y, h_Y, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_dX1, size_bytes);
  cudaMalloc((void**)&d_dX2, size_bytes);

  dim3 grid ((length + 255) / 256);
  dim3 block (256);

  // warmup
  for (int i = 0; i < repeat; i++) {
    MRCGradient <<<grid, block>>> (length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    MRCGradient2 <<<grid, block>>> (length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
  }

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient <<<grid, block>>> (length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    MRCGradient2 <<<grid, block>>> (length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  cudaMemcpy(h_dX1, d_dX1, size_bytes, cudaMemcpyDeviceToHost); 
  cudaMemcpy(h_dX2, d_dX2, size_bytes, cudaMemcpyDeviceToHost); 

  reference (length, h_Y, h_X1, h_X2, h_O, m, r_dX1, r_dX2);

  bool ok = true;
  for (int i = 0; i < length; i++) {
    if (fabs(h_dX1[i] - r_dX1[i]) > 1e-3 || fabs(h_dX2[i] - r_dX2[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_X1);
  cudaFree(d_X2);
  cudaFree(d_O);
  cudaFree(d_Y);
  cudaFree(d_dX1);
  cudaFree(d_dX2);

  free(h_X1);
  free(h_X2);
  free(h_O);
  free(h_Y);
  free(h_dX1);
  free(h_dX2);

  return 0;
}
