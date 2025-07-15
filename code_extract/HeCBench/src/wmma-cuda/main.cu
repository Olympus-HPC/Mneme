/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits> // std::numeric_limits
#include <string>
#include <vector>
#include <cuda.h>
#include <mma.h>

typedef half fp16;
typedef float fp32;

#include "reference.h"

using namespace nvcuda;

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(status)                                               \
  if (status != cudaSuccess) {                                                 \
    fprintf(stderr, "cuda error: '%s'(%d) at %s:%d\n",                         \
            cudaGetErrorString(status), status, __FILE__, __LINE__);           \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT *mat, uint32_t m, uint32_t n) {
  srand(m * n);
  auto ld = n;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      // Ascending order for each neighboring element.
      // Alternate sign for even / odd
      auto value = (i * n + j) % 13;
      mat[i * ld + j] =
          (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
    }
  }
}

// : 16 x 16
const int WMMA_M = 16;
const int WMMA_N = 16;

// : multiples of 16.
const int WMMA_K = 16;

// Device warp size
const uint32_t WAVE_SIZE = 32;

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: A workgroup will compute T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int NUM_WAVES_X = 4;
const int NUM_WAVES_Y = 4;
const int T_BLOCK_X = NUM_WAVES_X * WAVE_SIZE;
const int T_BLOCK_Y = NUM_WAVES_Y;

// The following device kernel is a naive implementation
// of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the M x N x K GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Note: demonstrate API usage in context of wave-level GEMM computation, and is not optimized.
__global__ void gemm(const uint32_t m, const uint32_t n, const uint32_t k,
                     fp16 const *__restrict__ a,
                     fp16 const *__restrict__ b,
                     fp32 const *c,
                     fp32 *d, const uint32_t lda, const uint32_t ldb,
                     const uint32_t ldc, const uint32_t ldd,
                     const fp32 alpha, const fp32 beta) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::row_major>();
  auto fragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::col_major>();
  auto fragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();
  auto fragAcc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();

  wmma::fill_fragment(fragAcc, 0.0f);

  // Tile using a 2D grid
  auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
  auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

  // Target C block
  auto cRow = majorWarp * WMMA_M;
  auto cCol = minorWarp * WMMA_N;

  // Bounds check
  if (cRow < m && cCol < n) {
    for (int i = 0; i < k; i += WMMA_K) {
      // Load the inputs
      wmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
      wmma::load_matrix_sync(fragB, b + (cCol * ldb + i), ldb);

      // Matrix multiply - accumulate using MFMA units
      wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
    }

    // Fetch C matrix
    wmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc,
                           wmma::mem_row_major);

    // D = alpha * A x B + beta * C
    for (int i = 0; i < fragC.num_elements; ++i) {
      fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
    }

    // Store to D
    wmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd,
                            wmma::mem_row_major);
  }
}

__host__ void gemm_wmma(uint32_t m, uint32_t n, uint32_t k, fp32 alpha,
                        fp32 beta, int32_t repeat, int32_t verify) {
  // Bounds check
  if ((m < (WMMA_M * NUM_WAVES_X) || n < (WMMA_N * NUM_WAVES_Y) ||
       k < WMMA_K) || (m % WMMA_M || n % WMMA_N || k % WMMA_K)) {
    std::cout << "Unsupported size!\n";
    return;
  }

  int lda = k; // row major
  int ldb = k; // col major
  int ldc = n; // row major
  int ldd = ldc;

  std::cout << "Initializing host data..." << std::endl;

  // Initialize input matrices
  std::vector<fp16> matrixA(m * k);
  std::vector<fp16> matrixB(k * n);
  std::vector<fp32> matrixC(m * n);

  // Fill outputs with NaN to catch contamination
  std::vector<fp32> matrixD(
      m * n, std::numeric_limits<fp32>::signaling_NaN());

  fill(matrixA.data(), m, k);
  fill(matrixB.data(), k, n);
  fill(matrixC.data(), m, n);

  std::cout << "Initializing device data..." << std::endl;

  // Allocate and copy device memory
  fp16 *d_a, *d_b;
  fp32 *d_c, *d_d;

  const size_t bytesA = matrixA.size() * sizeof(fp16);
  const size_t bytesB = matrixB.size() * sizeof(fp16);
  const size_t bytesC = matrixC.size() * sizeof(fp32);
  const size_t bytesD = matrixD.size() * sizeof(fp32);

  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytesA));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytesB));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytesC));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d, bytesD));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_a, matrixA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_b, matrixB.data(), bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_c, matrixC.data(), bytesC, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_d, matrixD.data(), bytesD, cudaMemcpyHostToDevice));

  auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
  auto gridDim = dim3((m + WMMA_M * NUM_WAVES_X - 1) / (WMMA_M * NUM_WAVES_X),
                      (n + WMMA_N * NUM_WAVES_Y - 1) / (WMMA_N * NUM_WAVES_Y));

  std::cout << "Launching GEMM kernel..." << std::endl;

  double elapsedTimeMs;

  std::chrono::time_point<std::chrono::steady_clock> start, end;

  for (int32_t w = 0; w <= repeat; w++) {
    if (w == 1) {
      cudaDeviceSynchronize();
      start = std::chrono::steady_clock::now();
    }

    gemm<<<gridDim, blockDim, 0, 0>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb,
                                      ldc, ldd, alpha, beta);

    if (w == repeat) {
      cudaDeviceSynchronize(); // throughput
      end = std::chrono::steady_clock::now();
      auto time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      elapsedTimeMs = time * 1e-6;
    }
  }

  // GEMM flops converge to 2*mnk
  auto gFlops = 2.0 * static_cast<double>(m) * n * k * 1.0e-9;
  auto tFlopsPerSec = gFlops * repeat / static_cast<double>(elapsedTimeMs);

  // Echo performance
  std::cout << "BlkM, BlkN, BlkK, "
            << "MatM, MatN, MatK, "
            << "alpha, lda, ldb, "
            << "beta, ldc, ldd, "
            << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

  std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", "
            << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
            << ", " << beta << ", " << ldc << ", " << ldd << ", "
            << elapsedTimeMs << ", " << gFlops << ", " << tFlopsPerSec
            << std::endl;

  if (verify) {
    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_CUDA_ERROR(
        cudaMemcpy(matrixD.data(), d_d, bytesD, cudaMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<fp32> matrixD_ref(
        m * n, std::numeric_limits<fp32>::signaling_NaN());
    gemm_cpu_h(m, n, k, matrixA.data(), matrixB.data(), matrixC.data(),
               matrixD_ref.data(), lda, ldb, ldc, ldd, alpha, beta);

    compareEqual<fp32>(matrixD.data(), matrixD_ref.data(), m * n);
  }

  // Release device memory
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  CHECK_CUDA_ERROR(cudaFree(d_d));

  std::cout << "Finished!" << std::endl;
}

void Usage(std::string program_name) {
  // Utility function to display argument usage
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << "<M> <N> <K> <repeat> <verify>\n\n";
  std::cout
      << "Dense matrix-matrix multiplication: D = alpha * (A * B) + beta * C\n";
  std::cout << "A: M * K, B: K * N, C: M * N, D: M * N\n";
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    Usage(argv[0]);
  }

  const uint32_t m = atoi(argv[1]);
  const uint32_t n = atoi(argv[2]);
  const uint32_t k = atoi(argv[3]);
  const int32_t repeat = atoi(argv[4]);
  const int32_t verify = atoi(argv[5]);
  gemm_wmma(m, n, k, 0.5f, 2.0f, repeat, verify);

  return 0;
}
