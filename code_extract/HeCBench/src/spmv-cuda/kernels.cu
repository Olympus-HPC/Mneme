#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cusparse.h>
#include "mv.h"

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

long reference(const int repeat,
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

  dim3 grids ((num_rows + 256 - 1) / 256);
  dim3 blocks (256);

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

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

long spmv_csr(const int repeat,
              const size_t num_rows,
              const REAL* x,
              const size_t nnz,
              REAL* matrix,
              REAL* y)
{
  size_t *row_indices = (size_t *) malloc((num_rows+1) * sizeof(size_t));
  size_t *col_indices = (size_t *) malloc(nnz * sizeof(size_t));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

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

  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  const cudaDataType DT = (sizeof(REAL) == 4) ? CUDA_R_32F : CUDA_R_64F;
 
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  REAL                 alpha      = 1.f;
  REAL                 beta       = 0.f;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_rows, nnz,
                                    d_row_indices, d_col_indices, d_values,
                                    CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                    CUSPARSE_INDEX_BASE_ZERO, DT) )
  // Create dense vector X
  CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, num_rows, d_x, DT) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, num_rows, d_y, DT) )
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                               handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY, DT,
                               CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
  cudaMalloc(&dBuffer, bufferSize);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, DT,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  free(values);
  free(row_indices);
  free(col_indices);

  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return time;
}

long spmv_coo(const int repeat,
              const size_t num_rows,
              const REAL* x,
              const size_t nnz,
              REAL* matrix,
              REAL* y)
{
  size_t *row_indices = (size_t *) malloc(nnz * sizeof(size_t));
  size_t *col_indices = (size_t *) malloc(nnz * sizeof(size_t));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize coo structure
  init_coo(row_indices, values, col_indices, matrix, num_rows, nnz);

  size_t *d_row_indices;
  size_t *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_indices, nnz*sizeof(size_t));
  cudaMalloc(&d_col_indices, nnz*sizeof(size_t));

  cudaMalloc(&d_values, nnz*sizeof(REAL));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row_indices, row_indices, nnz*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices, nnz*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);

  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  const cudaDataType DT = (sizeof(REAL) == 4) ? CUDA_R_32F : CUDA_R_64F;
 
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  REAL                 alpha      = 1.f;
  REAL                 beta       = 0.f;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  // Create sparse matrix A in COO format
  CHECK_CUSPARSE( cusparseCreateCoo(&matA, num_rows, num_rows, nnz,
                                    d_row_indices, d_col_indices, d_values,
                                    CUSPARSE_INDEX_64I,
                                    CUSPARSE_INDEX_BASE_ZERO, DT) )
  // Create dense vector X
  CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, num_rows, d_x, DT) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, num_rows, d_y, DT) )
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                               handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY, DT,
                               CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
  cudaMalloc(&dBuffer, bufferSize);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, DT,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  free(values);
  free(row_indices);
  free(col_indices);

  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return time;
}
