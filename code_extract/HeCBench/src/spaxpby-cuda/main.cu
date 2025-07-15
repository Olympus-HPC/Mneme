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

  if (argc != 5) {
    printf("The function computes the sum of a sparse vector and a dense vector "); 
    printf("in single-precision floating-point operations\n");
    printf("for i=0 to n-1        \n");
    printf("    Y[i] = beta * Y[i]\n");
    printf("for i=0 to nnz-1      \n");
    printf("    Y[X_indices[i]] += alpha * X_values[i]\n");
    printf("\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    printf("The size of the vector (n) is M * N\n");
    printf("nnz is the number of non-zero elements\n");
    return 1;
  }

  size_t m, n, nnz;

  m = atol(argv[1]);
  n = atol(argv[2]);
  nnz = atol(argv[3]);
  repeat = atoi(argv[4]);

  // Host problem definition
  const size_t size = m * n;

  const size_t value_size_bytes = nnz * sizeof(float);
  const size_t index_size_bytes = nnz * sizeof(size_t);
  const size_t size_bytes = size * sizeof(size_t);

  float *hA = (float*) malloc (size_bytes);
  float *hB = (float*) malloc (size_bytes); // the initial and final results computed on a host
  float *hY = (float*) malloc (size_bytes); // the results from a device
  float *hA_values = (float*) malloc (value_size_bytes);
  size_t *hA_indices = (size_t*) malloc (index_size_bytes);

  printf("Initializing input matrices..\n");
  init_matrix(hA, m, n, nnz);

  size_t k = 0;
  for (size_t i = 0; i < size; i++) {
    if (hA[i] != 0) {
      hA_indices[k] = i;
      hA_values[k] = hA[i];
      k++;
    }
  }

  init_matrix(hB, m, n, size);

  printf("Done\n");

  const float alpha = 1.0f;
  const float beta  = 1.0f;

  size_t *dX_indices;
  float *dY, *dX_values;
  CHECK_CUDA( cudaMalloc((void**) &dX_indices, index_size_bytes)  )
  CHECK_CUDA( cudaMalloc((void**) &dX_values,  value_size_bytes)  )
  CHECK_CUDA( cudaMalloc((void**) &dY,         size_bytes)  )

  CHECK_CUDA( cudaMemcpy(dX_indices, hA_indices, index_size_bytes,
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dX_values, hA_values, value_size_bytes,
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dY, hB, size_bytes,
                         cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // Create sparse vector X
  CHECK_CUSPARSE( cusparseCreateSpVec(&vecX, size, nnz, dX_indices, dX_values,
                                      CUSPARSE_INDEX_64I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F) )

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute Axpby
    CHECK_CUSPARSE( cusparseAxpby(handle, &alpha, vecX, &beta, vecY) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPAXPBY : %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpVec(vecX) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(hY, dY, size_bytes,
                         cudaMemcpyDeviceToHost) )

  printf("Computing the reference results..\n");
  for (int n = 0; n < repeat; n++) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      hB[i] = alpha * hA[i] + beta * hB[i];
    }
  }
  printf("Done\n");
  
  int correct = 1;
  for (size_t i = 0; i < size; i++) {
    if (fabsf(hY[i] - hB[i]) > 1e-2f) {
      correct = 0;
      break;
    }
  }
  if (correct)
      printf("axpby_example test PASSED\n");
  else
      printf("axpby_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA( cudaFree(dX_indices) )
  CHECK_CUDA( cudaFree(dX_values)  )
  CHECK_CUDA( cudaFree(dY) )
  free(hA);
  free(hB);
  free(hY);
  free(hA_values);
  free(hA_indices);
  return EXIT_SUCCESS;
}
