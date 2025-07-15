#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "mv.h"

// sparse matrix vector multiply using the CSR format
__global__ void mv_csr(const size_t num_rows,
                       const size_t *row_indices,
                       const size_t *col_indices,
                       const REAL *values,
                       const REAL *x,
                             REAL *y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    size_t row_start = row_indices[i];
    size_t row_end = row_indices[i+1];

    REAL temp = 0;
    for(size_t n = row_start; n < row_end; n++){
      temp += values[n] * x[col_indices[n]];
    }
    y[i] = temp;
  }
}

// vector sparse matrix vector multiply using the CSR format
template <int BS>
__global__ void vector_mv_csr(const size_t num_rows,
                              const size_t *row_indices,
                              const size_t *col_indices,
                              const REAL *values,
                              const REAL *x,
                                    REAL *y)
{
  size_t m = blockIdx.x * blockDim.y + threadIdx.y;
  if (m < num_rows) {
    size_t row_start = row_indices[m];
    size_t row_end = row_indices[m+1];

    REAL temp = 0;
    for(size_t n = row_start + threadIdx.x; n < row_end; n += BS){
      temp += values[n] * x[col_indices[n]];
    }
    #pragma unroll
    for (int i = BS >> 1; i > 0; i >>= 1)
      temp += __shfl_down_sync(0xFFFFFFFF, temp, i, BS);

    y[m] = temp;
  }
}


// dense matrix vector multiply
__global__ void mv_dense(const size_t num_rows, const REAL* matrix, const REAL* x, REAL* y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    REAL temp = 0;
    for (size_t j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != (REAL)0) 
        temp += matrix[i * num_rows + j] * x[j];
    }
    y[i] = temp;
  }
}

long mv_dense_parallel(const int repeat,
                       const int bs,
                       const size_t num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  REAL *d_x, *d_matrix, *d_y;
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), cudaMemcpyHostToDevice);

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mv_dense<<<grids, blocks>>>(num_rows, d_matrix, d_x, d_y);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_matrix);

  return time;
}

long mv_csr_parallel(const int repeat,
                     const int bs,
                     const size_t num_rows,
                     const size_t *row_indices,
                     const size_t *col_indices,
                     const REAL* values,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  size_t *d_row_indices;
  size_t *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  cudaMalloc(&d_col_indices, nnz*sizeof(size_t));
  cudaMalloc(&d_values, nnz*sizeof(REAL));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices, nnz*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mv_csr<<<grids, blocks>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return time;
}

// Reference
// https://github.com/ROCm/rocm-blogs/blob/release/blogs/high-performance-computing/spmv/part-1/examples/vector_csr.cpp
size_t prevPowerOf2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

long vector_mv_csr_parallel(const int repeat,
                            const int bs,
                            const size_t num_rows,
                            const size_t *row_indices,
                            const size_t *col_indices,
                            const REAL* values,
                            const REAL* x,
                            const size_t nnz,
                            REAL* matrix,
                            REAL* y)
{
  size_t *d_row_indices;
  size_t *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  cudaMalloc(&d_col_indices, nnz*sizeof(size_t));
  cudaMalloc(&d_values, nnz*sizeof(REAL));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices, nnz*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);

  int nnz_per_row = nnz / num_rows;
  int threads_per_row = prevPowerOf2(nnz_per_row);
  int warpSize = 32;
  // limit the number of threads per row to be no larger than the wavefront (warp) size
  threads_per_row = threads_per_row > warpSize ? warpSize : threads_per_row;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks = (num_rows + rows_per_block - 1) / rows_per_block;

  dim3 grid(num_blocks, 1, 1);
  dim3 block(threads_per_row, rows_per_block, 1);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if (threads_per_row <= 2)
      vector_mv_csr<2><<<grid, block>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
    else if (threads_per_row <= 4)
      vector_mv_csr<4><<<grid, block>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
    else if (threads_per_row <= 8)
      vector_mv_csr<8><<<grid, block>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
    else if (threads_per_row <= 16)
      vector_mv_csr<16><<<grid, block>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
    else if (threads_per_row <= 32)
      vector_mv_csr<32><<<grid, block>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return time;
}
