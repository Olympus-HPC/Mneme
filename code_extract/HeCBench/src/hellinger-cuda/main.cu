#include <iostream>
#include <new>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define BLOCK_SIZE 16

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif


/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 768 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#ifdef VERIFY
#include "verify.h"
#endif

__global__ 
void hellinger(
  const FP *__restrict__ a, 
  const FP *__restrict__ b, 
        FP *__restrict__ c, 
  const int m, const int n, const int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if( col < k && row < m)
    {
        FP sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += SQRT(a[row * n + i] * b[i * k + col]);
        }
        const FP value = (FP)1.0 - sum;
        const FP gate = (!signbit(value));
        c[row * k + col] = SQRT(gate * value);
    }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  // 2D arrays on host side.
  FP(*a_host)[N] = new FP[M][N];
  FP(*b_host)[P] = new FP[N][P];
  // host-side cpu result
  FP(*c_host)[P] = new FP[M][P];
  // host-side gpu result
  FP(*c_back)[P] = new FP[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = (FP)1.0 / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    FP sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  FP *a_device, *b_device, *c_device;

  cudaMalloc((void **) &a_device, sizeof(FP)*M*N);
  cudaMalloc((void **) &b_device, sizeof(FP)*N*P);
  cudaMalloc((void **) &c_device, sizeof(FP)*M*P);

  cudaMemcpy(a_device, a_host, sizeof(FP)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, b_host, sizeof(FP)*N*P, cudaMemcpyHostToDevice);

  unsigned int grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hellinger<<<dimGrid, dimBlock>>>(a_device, b_device, c_device, M, N, P);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)\n";

  cudaMemcpy(c_back, c_device, sizeof(int)*M*P, cudaMemcpyDeviceToHost);

  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
            << ") * b(" << N << "," << P << ")\n";

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);
  return 0;
}
