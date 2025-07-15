#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparse<t>nnz
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

  /*
    https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-t-nnz
    cusparseStatus_t
    cusparseSnnz(cusparseHandle_t     handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const float*             A,
             int                      lda,
             int*                     nnzPerRowColumn,
             int*                     nnzTotalDevHostPtr)
   */

  if (argc != 5) {
    printf("This function computes the number of nonzero elements per row or column");
    printf(" and the total number of nonzero elements in a dense matrix.\n");
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
  const int64_t lda = num_cols;
  const int64_t dense_size = m * n;
  const int64_t dense_size_bytes = dense_size * sizeof(float);
  const int64_t nnzRowCol_size_bytes = num_rows * sizeof(int);

  float *h_dense = (float*) malloc (dense_size_bytes);

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);

  // device results
  int* nnzPerRowColumn = (int*) malloc (nnzRowCol_size_bytes);

  // allocate the device arrays
  int *d_nnzPerRowColumn;
  float *d_dense;
  CHECK_CUDA( cudaMalloc((void**) &d_nnzPerRowColumn, nnzRowCol_size_bytes)) 
  CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size_bytes))
  CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size_bytes,
                         cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t   handle = NULL;
  cusparseMatDescr_t descr = 0;

  int nnzTotal = 0;

  CHECK_CUSPARSE( cusparseCreate(&handle) )
  CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )
  CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    CHECK_CUSPARSE( cusparseSnnz(handle,
                                 CUSPARSE_DIRECTION_COLUMN,
                                 num_rows,
                                 num_cols,
                                 descr,
                                 d_dense,
                                 lda,
                                 d_nnzPerRowColumn,
                                 &nnzTotal) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of cusparseSnnz : %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(nnzPerRowColumn, d_nnzPerRowColumn, nnzRowCol_size_bytes,
                         cudaMemcpyDeviceToHost) )
  int correct = 1; 
  if (h_nnz != nnzTotal) {
    printf("nnz: %d != %d\n", (int)h_nnz, nnzTotal);
    correct = 0;
    goto print_error;
  }

  for (int64_t i = 0; i < num_rows; i++) {
    int nnz = 0; // nnz per row
    for (int64_t j = 0; j < num_cols; j++) {
      if (h_dense[i*num_cols+j] != 0) nnz++; 
    }
    if (nnz != nnzPerRowColumn[i]) {
      printf("@row %ld %d != %d\n", i, nnz, nnzPerRowColumn[i]);
      correct = 0;
      goto print_error;
    }
  }

  print_error:
  if (correct)
      printf("sparse_nnz_example test PASSED\n");
  else
      printf("sparse_nnz_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA( cudaFree(d_nnzPerRowColumn) )
  CHECK_CUDA( cudaFree(d_dense) )
  free(h_dense);
  free(nnzPerRowColumn);
  return EXIT_SUCCESS;
}
