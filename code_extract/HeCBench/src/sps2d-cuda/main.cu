/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include "utils.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
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

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 5) {
    printf("The function converts a sparse matrix into a MxN dense matrix\n");
    printf("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    printf("nnz is the number of non-zero elements\n");
    return 1;
  }

  int64_t m, n, h_nnz;

  m = atol(argv[1]);
  n = atol(argv[2]);
  h_nnz = atol(argv[3]);
  repeat = atoi(argv[4]);

  // Host problem definition
  const int64_t num_rows = m;
  const int64_t num_cols = n;
  const int64_t dense_size = m * n;

  const int64_t dense_size_bytes  = dense_size * sizeof(float);
  const int64_t value_size_bytes  = h_nnz * sizeof(float);
  const int64_t colidx_size_bytes = h_nnz * sizeof(int64_t);
  const int64_t rowidx_size_bytes = (num_rows + 1) * sizeof(int64_t);

  // reference results
  float *h_dense = (float*) malloc (dense_size_bytes);

  // device results
  float *h_dense_result = (float*) malloc (dense_size_bytes);

  float *h_csr_values = (float*) malloc (value_size_bytes);
  int64_t *h_csr_columns = (int64_t*) malloc (colidx_size_bytes);
  int64_t *h_csr_offsets = (int64_t*) malloc (rowidx_size_bytes);

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);

  init_csr(h_csr_offsets, h_csr_values, h_csr_columns,
           h_dense, num_rows, num_cols, h_nnz);
  //--------------------------------------------------------------------------
  // Device memory management
  int64_t *d_csr_offsets, *d_csr_columns;
  float *d_csr_values, *d_dense;

  CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size_bytes))
  CHECK_CUDA( cudaMemset(d_dense, 0, dense_size_bytes))

  // allocate CSR column indices and values
  CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, h_nnz * sizeof(int64_t)) )
  CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  h_nnz * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                         (num_rows + 1) * sizeof(int64_t)) )

  CHECK_CUDA( cudaMemcpy(d_csr_offsets, h_csr_offsets,
                         (num_rows + 1) * sizeof(int64_t),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_csr_columns, h_csr_columns, h_nnz * sizeof(int64_t),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_csr_values, h_csr_values, h_nnz * sizeof(float),
                         cudaMemcpyHostToDevice) )

  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, h_nnz,
                                    d_csr_offsets, d_csr_columns, d_csr_values,
                                    CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

  // Create dense matrix B
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, num_rows, num_cols, num_cols, d_dense,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )

  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSparseToDense_bufferSize(
                                      handle, matA, matB,
                                      CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                      &bufferSize) )
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseSparseToDense(handle, matA, matB,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SparseToDense_convert : %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check

  CHECK_CUDA( cudaMemcpy(h_dense_result, d_dense, dense_size_bytes,
                         cudaMemcpyDeviceToHost) )

  int correct = 1; 
  int64_t nnz = 0;
  for (int64_t i = 0; i < dense_size; i++) {
    if (h_dense_result[i] != 0) nnz++;

    if (h_dense[i] != h_dense_result[i]) {
      printf("@%ld: %f != %f\n", i, h_dense[i], h_dense_result[i]);
      correct = 0;
      goto print_error;
    }
  }

  if (nnz != h_nnz) {
    printf("nnz: %ld != %ld\n", h_nnz, nnz);
    correct = 0;
  }

  print_error:
  if (correct)
      printf("sparse2dense_csr_example test PASSED\n");
  else                    
      printf("sparse2dense_csr_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(d_csr_offsets) )
  CHECK_CUDA( cudaFree(d_csr_columns) )
  CHECK_CUDA( cudaFree(d_csr_values) )
  CHECK_CUDA( cudaFree(d_dense) )
  free(h_dense);
  free(h_dense_result);
  free(h_csr_values);
  free(h_csr_columns);
  free(h_csr_offsets);
  return EXIT_SUCCESS;
}
