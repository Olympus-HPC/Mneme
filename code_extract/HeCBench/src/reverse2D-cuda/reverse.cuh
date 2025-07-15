/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define DI  inline __device__
#define HDI inline __host__ __device__

#include <chrono>
#include <cuda.h>
#include "vectorized.cuh"

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
  return (a + b - 1) / b;
}

template <typename T>
HDI void swapVals(T& a, T& b)
{
  T tmp = a;
  a     = b;
  b     = tmp;
}


template <typename data_t, int veclen_>
__global__
void reverseKernel(data_t* out,
                   const data_t* in,
                   int nrows,
                   int ncols,
                   bool rowMajor,
                   bool alongRows,
                   int len)
{
  typedef TxN_t<data_t, veclen_> VecType;
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * VecType::Ratio;
  if (idx >= len) return;
  int srcIdx, dstIdx;
  if (!rowMajor && !alongRows) {
    int srcRow = idx % nrows;
    int srcCol = idx / nrows;
    int dstRow = srcRow;
    int dstCol = ncols - srcCol - 1;
    srcIdx     = idx;
    dstIdx     = dstCol * nrows + dstRow;
  } else if (!rowMajor && alongRows) {
    int mod    = ceildiv(nrows, 2);
    int srcRow = idx % mod;
    int srcCol = idx / mod;
    int dstRow = nrows - srcRow - VecType::Ratio;
    int dstCol = srcCol;
    srcIdx     = srcCol * nrows + srcRow;
    dstIdx     = dstCol * nrows + dstRow;
  } else if (rowMajor && !alongRows) {
    int mod    = ceildiv(ncols, 2);
    int srcRow = idx / mod;
    int srcCol = idx % mod;
    int dstRow = srcRow;
    int dstCol = ncols - srcCol - VecType::Ratio;
    srcIdx     = srcCol + srcRow * ncols;
    dstIdx     = dstCol + dstRow * ncols;
  } else {
    int srcRow = idx / ncols;
    int srcCol = idx % ncols;
    int dstRow = nrows - srcRow - 1;
    int dstCol = srcCol;
    srcIdx     = idx;
    dstIdx     = dstCol + dstRow * ncols;
  }
  VecType a, b;
  a.load(in, srcIdx);
  b.load(in, dstIdx);
  // while reversing along coalesced dimension, also reverse the elements
  if ((rowMajor && !alongRows) || (!rowMajor && alongRows)) {
    #pragma unroll
    for (int i = 0; i < VecType::Ratio; ++i) {
      swapVals(a.val.data[i], a.val.data[VecType::Ratio - i - 1]);
      swapVals(b.val.data[i], b.val.data[VecType::Ratio - i - 1]);
    }
  }
  a.store(out, dstIdx);
  b.store(out, srcIdx);
}

template <typename data_t, int veclen_, int TPB>
long reverseImpl(data_t* out,
                 const data_t* in,
                 int nrows,
                 int ncols,
                 bool rowMajor,
                 bool alongRows,
                 cudaStream_t stream)
{
  int len         = alongRows ? ceildiv(nrows, 2) * ncols : nrows * ceildiv(ncols, 2);
  const int nblks = ceildiv(veclen_ ? len / veclen_ : len, TPB);

  auto start = std::chrono::steady_clock::now();

  reverseKernel<data_t, veclen_>
    <<<nblks, TPB, 0, stream>>>(out, in, nrows, ncols, rowMajor, alongRows, len);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Reversal of the input matrix along the specified dimension
 * @tparam data_t data-type upon which the math operation will be performed
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output matrix (supports inplace operation)
 * @param in the input matrix
 * @param nrows number of rows in the input matrix
 * @param ncols number of cols in the input matrix
 * @param rowMajor input matrix is row major or not
 * @param alongRows whether to reverse along rows or not
 * @param stream cuda stream where to launch work
 */
template <typename data_t, int TPB = 256>
long reverse(data_t* out,
             const data_t* in,
             int nrows,
             int ncols,
             bool rowMajor,
             bool alongRows,
             cudaStream_t stream)
{
  size_t bytes = (rowMajor ? ncols : nrows) * sizeof(data_t);
  long time;
  if (16 / sizeof(data_t) && bytes % 16 == 0) {
    time = reverseImpl<data_t, 16 / sizeof(data_t), TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, stream);
  } else if (8 / sizeof(data_t) && bytes % 8 == 0) {
    time = reverseImpl<data_t, 8 / sizeof(data_t), TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, stream);
  } else if (4 / sizeof(data_t) && bytes % 4 == 0) {
    time = reverseImpl<data_t, 4 / sizeof(data_t), TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, stream);
  } else if (2 / sizeof(data_t) && bytes % 2 == 0) {
    time = reverseImpl<data_t, 2 / sizeof(data_t), TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, stream);
  } else if (1 / sizeof(data_t)) {
    time = reverseImpl<data_t, 1 / sizeof(data_t), TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, stream);
  } else {
    time = reverseImpl<data_t, 1, TPB>(out, in, nrows, ncols, rowMajor, alongRows, stream);
  }
  return time;
}

