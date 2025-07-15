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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cusparse.h>
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

  if (argc != 8) {
    printf("Single-precision sparse matrix-matrix multiplication into sparse matrix,\n");
    printf("where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <B_nnz> <repeat> <verify>\n", argv[0]);
    printf("SPMM (A, B, C) where (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }

  int m, k, n, a_nnz, b_nnz, verify;

  m = atoi(argv[1]);
  k = atoi(argv[2]);
  n = atoi(argv[3]);
  a_nnz = atoi(argv[4]);
  b_nnz = atoi(argv[5]);
  repeat = atoi(argv[6]);
  verify = atoi(argv[7]);

  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = n;
  const int B_nnz      = b_nnz;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(size_t);

  const size_t B_value_size_bytes  = B_nnz * sizeof(float);
  const size_t B_colidx_size_bytes = B_nnz * sizeof(int);
  const size_t B_rowidx_size_bytes = (B_num_rows + 1) * sizeof(size_t);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  float *hB_values = (float*) malloc (B_value_size_bytes);
  int *hB_columns = (int*) malloc (B_colidx_size_bytes);
  int *hB_offsets = (int*) malloc (B_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hB, B_num_rows, B_num_cols, B_nnz);
  init_csr(hB_offsets, hB_values, hB_columns, hB,
           B_num_rows, B_num_cols, B_nnz);

  float               alpha       = 1.0f;
  float               beta        = 0.0f;
  cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType        computeType = CUDA_R_32F;

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_offsets, *dA_columns,
        *dB_offsets, *dB_columns,
        *dC_offsets, *dC_columns;
  float *dA_values, *dB_values, *dC_values;
  // allocate A
  CHECK_CUDA( cudaMalloc((void**) &dA_offsets,
                         (A_num_rows + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
  // allocate B
  CHECK_CUDA( cudaMalloc((void**) &dB_offsets,
                         (B_num_rows + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )
  // allocate C offsets
  CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                         (A_num_rows + 1) * sizeof(int)) )

  // copy A
  CHECK_CUDA( cudaMemcpy(dA_offsets, hA_offsets,
                         (A_num_rows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                         A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
  // copy B
  CHECK_CUDA( cudaMemcpy(dB_offsets, hB_offsets,
                         (B_num_rows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                         B_nnz * sizeof(float), cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void*  dBuffer1    = NULL, *dBuffer2   = NULL;
  size_t bufferSize1 = 0,    bufferSize2 = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_offsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                    dB_offsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  // Algorithm 1
  // 
  // Invokes cusparseSpGEMM_compute twice. The first invocation provides an upper bound of the memory required for the computation.
  // 
  // The required memory is generally several times larger of the actual memory used.
  // 
  // The user can provide an arbitrary buffer size bufferSize2 in the second invocation. If it is not sufficient, the routine will returns CUSPARSE_STATUS_INSUFFICIENT_RESOURCES status.
  // 
  // Provides better performance than other algorithms.
  // 
  // Provides deterministic (bit-wise) results for each run.

  cusparseSpGEMMDescr_t spgemmDesc;
  CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

  // ask bufferSize1 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, NULL) )
  CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1) )

  // ask bufferSize2 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_compute(handle, opA, opB,
                             &alpha, matA, matB, &beta, matC,
                             computeType, CUSPARSE_SPGEMM_DEFAULT,
                             spgemmDesc, &bufferSize2, NULL) )
  CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPMM compute: %f (us)\n", (time * 1e-3f) / repeat);

  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                       &C_nnz1) )
  // allocate the CSR structures for the matrix C
  CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
  CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of cusparseSpGEMM_copy

  // update matC with the new pointers
  CHECK_CUSPARSE(
      cusparseCsrSetPointers(matC, dC_offsets, dC_columns, dC_values) )

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CHECK_CUSPARSE(
      cusparseSpGEMM_copy(handle, opA, opB,
                          &alpha, matA, matB, &beta, matC,
                          computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference SPMM results..\n");
    float *hC_values;
    int *hC_columns;
    int *hC_offsets;
    int C_nnz = spmm (hA, hB, 
                      hC_values,  hC_offsets, hC_columns,
                      A_num_cols, A_num_rows, B_num_cols);

    int *hC_offsets_tmp = (int*) malloc ((A_num_rows + 1) * sizeof(int));
    int *hC_columns_tmp = (int*) malloc (C_nnz * sizeof(int));
    float *hC_values_tmp = (float*) malloc (C_nnz * sizeof(float));

    CHECK_CUDA( cudaMemcpy(hC_offsets_tmp, dC_offsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows + 1; i++) {
      if (hC_offsets_tmp[i] != hC_offsets[i]) {
        correct = 0;
        break;
      }
    }
    for (int i = 0; i < C_nnz; i++) {
      if (hC_columns_tmp[i] != hC_columns[i] ||
          fabsf(hC_values_tmp[i] - hC_values[i]) > 1e-2f) {
        correct = 0;                       
        break;
      }
    }

    free(hC_values);
    free(hC_columns);
    free(hC_offsets);
    free(hC_values_tmp);
    free(hC_columns_tmp);
    free(hC_offsets_tmp);

    if (correct)
        printf("spgemm_example test PASSED\n");
    else {
        printf("spgemm_example test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA( cudaFree(dBuffer1) )
  CHECK_CUDA( cudaFree(dBuffer2) )
  CHECK_CUDA( cudaFree(dA_offsets) )
  CHECK_CUDA( cudaFree(dA_columns) )
  CHECK_CUDA( cudaFree(dA_values) )
  CHECK_CUDA( cudaFree(dB_offsets) )
  CHECK_CUDA( cudaFree(dB_columns) )
  CHECK_CUDA( cudaFree(dB_values) )
  CHECK_CUDA( cudaFree(dC_offsets) )
  CHECK_CUDA( cudaFree(dC_columns) )
  CHECK_CUDA( cudaFree(dC_values) )
  free(hA);
  free(hB);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  free(hB_values);
  free(hB_columns);
  free(hB_offsets);
  return EXIT_SUCCESS;
}
