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

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>
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

template <typename fp, typename intType>
int sort_sparse_matrix(intType a_nrows, intType a_ncols,
                       intType a_nnz, int repeat)
{
  fp *m = (fp*) malloc (sizeof(fp) * a_nrows * a_ncols);
  init_matrix(m, a_nrows, a_ncols, a_nnz);

  intType *ia = (intType*) malloc ((a_nrows+1) * sizeof(intType));
  intType *ja = (intType*) malloc (a_nnz * sizeof(intType));
  fp *a = (fp*) malloc (a_nnz * sizeof(fp));

  // reference results
  init_csr(ia, a, ja, m, a_nrows, a_ncols, a_nnz);

  // device results
  intType *ia_result = (intType*) malloc ((a_nrows+1) * sizeof(intType));
  intType *ja_result = (intType*) malloc (a_nnz * sizeof(intType));
  fp *a_result = (fp*) malloc (a_nnz * sizeof(fp));

  // shuffle the inputs
  memcpy(ia_result, ia, (a_nrows+1) * sizeof(intType));
  memcpy(ja_result, ja, a_nnz * sizeof(intType));
  memcpy(a_result, a, a_nnz * sizeof(fp));
  shuffle_matrix_data<fp, intType>(ia_result, ja_result, a_result, a_nrows, a_nnz);

  //--------------------------------------------------------------------------
  // Device memory management
  intType    *a_rowptr, *a_colind, *d_permutation;
  fp *a_val, *a_val_sorted;
  void   *d_buffer;
  size_t bufferSize;
  CHECK_CUDA( cudaMalloc((void**) &a_rowptr,      (a_nrows+1) * sizeof(intType)) )
  CHECK_CUDA( cudaMalloc((void**) &a_colind,      a_nnz * sizeof(intType)) )
  CHECK_CUDA( cudaMalloc((void**) &a_val,         a_nnz * sizeof(fp)) )
  CHECK_CUDA( cudaMalloc((void**) &a_val_sorted,  a_nnz * sizeof(fp)) )
  CHECK_CUDA( cudaMalloc((void**) &d_permutation, a_nnz * sizeof(intType)) )

  CHECK_CUDA( cudaMemcpy(a_rowptr, ia_result, (a_nrows+1) * sizeof(intType),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(a_colind, ja_result, a_nnz * sizeof(intType),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(a_val, a_result, a_nnz * sizeof(fp),
                         cudaMemcpyHostToDevice) )

  std::cout << "\n\t\tBasic info of the sparse matrix:\n";

  const fp ave_nnz_per_row = static_cast<fp>(a_nnz)/a_nrows;
  std::cout << "\t\t\tA_nrows = A_ncols = " << a_nrows << std::endl;
  std::cout << "\t\t\tA_nnz   = " << a_nnz << std::endl;

  intType max_nnz_per_row = 0;
  intType min_nnz_per_row = a_nnz;
  for (intType row = 0; row < a_nrows; ++row) {
    const intType loc_nnz_per_row = ia[row+1] - ia[row];

    if (loc_nnz_per_row > max_nnz_per_row) 
      max_nnz_per_row = loc_nnz_per_row;

    if (loc_nnz_per_row < min_nnz_per_row) 
      min_nnz_per_row = loc_nnz_per_row;
  }

  std::cout << "\t\t\t\tmin_nnz_per_row = " << min_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tave_nnz_per_row = " << ave_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tmax_nnz_per_row = " << max_nnz_per_row << std::endl;
  std::cout << std::endl;
  
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  cusparseMatDescr_t descr = 0;
  CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )

  CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )

  // Query working space of CSR sort
  CHECK_CUSPARSE( cusparseXcsrsort_bufferSizeExt(handle, a_nrows,
                                                 a_ncols, a_nnz, a_rowptr,
                                                 a_colind, &bufferSize) )
  CHECK_CUDA( cudaMalloc(&d_buffer, bufferSize) )

  // Setup permutation vector to identity
  CHECK_CUDA( cudaMalloc((void**)&d_permutation, sizeof(intType)*a_nnz) )
  CHECK_CUSPARSE( cusparseCreateIdentityPermutation(handle, a_nnz,
                                                    d_permutation) )

  // Deprecated cusparseDgthr and cusparseSgthr replaced with cusparseGather
  cusparseSpVecDescr_t vec_permutation;
  cusparseDnVecDescr_t vec_values;

  cudaDataType valueType;
  if constexpr (std::is_same_v<fp, double>) // C++17
    valueType = CUDA_R_64F;
  else
    valueType = CUDA_R_32F;

  // Create sparse vector for the permutation
  CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, a_nnz, a_nnz,
                                      d_permutation, a_val_sorted,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, valueType) )

  // Create dense vector for the csr values
  CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, a_nnz, a_val, valueType) )

  CHECK_CUDA( cudaDeviceSynchronize() )
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    CHECK_CUSPARSE( cusparseXcsrsort(handle, a_nrows, a_ncols, a_nnz, descr,
                                     a_rowptr, a_colind, d_permutation, d_buffer) )

    CHECK_CUSPARSE( cusparseGather(handle, vec_values, vec_permutation) )

  }

  CHECK_CUDA( cudaDeviceSynchronize() )
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of CSR sort: " << (time * 1e-3f) / repeat
            << " us" << std::endl;

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) )
  CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(ia_result, a_rowptr, (a_nrows+1) * sizeof(intType),
                         cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaMemcpy(ja_result, a_colind, a_nnz * sizeof(intType),
                         cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaMemcpy(a_result, a_val_sorted, a_nnz * sizeof(fp),
                         cudaMemcpyDeviceToHost) )

  int error;

  error = memcmp(ia_result, ia, (a_nrows+1) * sizeof(intType));
  if (error) {
    std::cout << "Error: row index arrays mismatch" << std::endl;
    goto print_error;
  }

  error = memcmp(ja_result, ja, a_nnz * sizeof(intType));
  if (error) {
    std::cout << "Error: column index arrays mismatch" << std::endl;
    goto print_error;
  }

  error = memcmp(a_result, a, a_nnz * sizeof(fp));
  if (error) {
    std::cout << "Error: value arrays mismatch" << std::endl;
    goto print_error;
  }

  print_error:
  if (error == 0)
      std::cout << "csrsort_example test PASSED" << std::endl;
  else
      std::cout << "csrsort_example test FAILED: wrong result" << std::endl;
  //--------------------------------------------------------------------------
  // memory deallocation
  CHECK_CUDA( cudaFree(a_rowptr) )
  CHECK_CUDA( cudaFree(a_colind) )
  CHECK_CUDA( cudaFree(d_permutation) )
  CHECK_CUDA( cudaFree(a_val) )
  CHECK_CUDA( cudaFree(a_val_sorted) )
  CHECK_CUDA( cudaFree(d_buffer) )

  free(m);
  free(ia);
  free(ja);
  free(a);
  free(ia_result);
  free(ja_result);
  free(a_result);
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 5) {
    std::cout << "The function sorts the column indices of each row in a sparse MxN matrix" << std::endl;
    std::cout << "The sparse matrix is represented in CSR (Compressed Sparse Row) storage format" << std::endl;
    std::cout << "Usage " << argv[0] << " <M> <N> <nnz> <repeat>" << std::endl;
    std::cout << "nnz is the number of non-zero elements" << std::endl;
    return 1;
  }

  int m, n, nnz;

  m = atoi(argv[1]);
  n = atoi(argv[2]);
  nnz = atoi(argv[3]);
  repeat = atoi(argv[4]);

  int status = sort_sparse_matrix<float, int>(m, n, nnz, repeat);
  if (status == EXIT_FAILURE) return status;

  return status;
}
