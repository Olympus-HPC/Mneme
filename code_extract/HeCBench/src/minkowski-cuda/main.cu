#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>
#include <cuda.h>

using namespace std;

#define BLOCK_SIZE 16

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 512 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

#include "verify.cpp"

template <int m, int n, int k>
__global__ 
void minkowski(
  const float *__restrict__ a,
  const float *__restrict__ b,
        float *__restrict__ c,
  const float p,
  const float one_over_p)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if( col < k && row < m)
  {
    float sum = 0;
    for(int i = 0; i < n; i++)
    {
      sum += powf(fabsf(a[row * n + i] - b[i * k + col]), p);
    }
    c[row * k + col] = powf(sum, one_over_p);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[K] = new float[N][K];
  // host-side cpu result
  float(*c_host)[K] = new float[M][K];
  // host-side gpu result
  float(*c_back)[K] = new float[M][K];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < K; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < K; j++) { 
    float sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  float *a_device, *b_device, *c_device;

  cudaMalloc((void **) &a_device, sizeof(float)*M*N);
  cudaMalloc((void **) &b_device, sizeof(float)*N*K);
  cudaMalloc((void **) &c_device, sizeof(float)*M*K);

  cudaMemcpy(a_device, a_host, sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, b_host, sizeof(float)*N*K, cudaMemcpyHostToDevice);

  unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  cout << "Problem size: c(" << M << "," << K << ") = a(" << M << "," << N
       << ") * b(" << N << "," << K << ")\n";

  for (int k = 1; k <= 4; k++) {
    printf("Minkowski distance with p = %d\n", k);
    const float p = (float)k;
    const float one_over_p = 1.f / p;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      minkowski<M, N, K><<<dimGrid, dimBlock>>>(a_device, b_device, c_device, p, one_over_p);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    cudaMemcpy(c_back, c_device, sizeof(int)*M*K, cudaMemcpyDeviceToHost);

    #ifdef VERIFY
    VerifyResult(a_host, b_host, c_host, c_back, p, one_over_p);
    #endif
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);
  return 0;
}
