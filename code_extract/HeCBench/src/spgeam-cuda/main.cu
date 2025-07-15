#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

  if (argc != 6) {
    printf("The function performs an out-of-place transpose of a sparse matrix into a new one,");
    printf("where the input sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <M> <K> <nnz> <repeat> <verify>\n", argv[0]);
    printf("Input matrix A: M * K\n");
    printf("Output matrix B: K * M\n");
    return 1;
  }

  int m, k, a_nnz, verify;

  m = atoi(argv[1]);
  k = atoi(argv[2]);
  a_nnz = atoi(argv[3]);
  repeat = atoi(argv[4]);
  verify = atoi(argv[5]);

  const int transpose = 1;

  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = A_num_rows;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;

  float *hA = (float*) malloc (A_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_offsets, // row ptr
        *dA_columns, // column indices
        *dB_offsets, // col ptr
        *dB_rows;    // row indices
  float *dA_values, *dB_values;

  // allocate A
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_offsets, (A_num_rows + 1) * sizeof(int)) )

  // allocate B 
  const size_t B_value_size_bytes  = B_nnz * sizeof(float);
  const size_t B_rowidx_size_bytes = B_nnz * sizeof(int);
  const size_t B_colidx_size_bytes = (A_num_cols+1) * sizeof(int);

  CHECK_CUDA( cudaMalloc((void**) &dB_values, B_value_size_bytes) )
  CHECK_CUDA( cudaMalloc((void**) &dB_rows, B_rowidx_size_bytes) )
  CHECK_CUDA( cudaMalloc((void**) &dB_offsets, B_colidx_size_bytes) )

  // copy A
  CHECK_CUDA( cudaMemcpy(dA_offsets, hA_offsets, A_rowidx_size_bytes,
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_colidx_size_bytes,
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_value_size_bytes,
                         cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle     = NULL;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
        handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_offsets, dA_columns,
        dB_values, dB_offsets, dB_rows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize) )
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    CHECK_CUSPARSE(cusparseCsr2cscEx2(
        handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_offsets, dA_columns,
        dB_values, dB_offsets, dB_rows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, dBuffer))
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of cuSparse csr2cscEx2 : %f (us)\n", (time * 1e-3f) / repeat);

  CHECK_CUSPARSE( cusparseDestroy(handle) )

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference results..\n");
    float *hB = (float*) malloc (B_size * sizeof(float));
    transpose_dense (hA, hB, A_num_cols, A_num_rows, transpose);

    float *hB_values = (float*) malloc (B_value_size_bytes);
    int *hB_rows = (int*) malloc (B_rowidx_size_bytes);
    int *hB_offsets = (int*) malloc (B_colidx_size_bytes);
    float *hB_values_tmp = (float*) malloc (B_value_size_bytes);
    int *hB_rows_tmp = (int*) malloc (B_rowidx_size_bytes);
    int *hB_offsets_tmp = (int*) malloc (B_colidx_size_bytes);

    init_csr(hB_offsets, hB_values, hB_rows, hB,
             B_num_rows, B_num_cols, B_nnz);

    CHECK_CUDA( cudaMemcpy(hB_offsets_tmp, dB_offsets, B_colidx_size_bytes,
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hB_rows_tmp, dB_rows, B_rowidx_size_bytes,
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hB_values_tmp, dB_values, B_value_size_bytes,
                           cudaMemcpyDeviceToHost) )

    int correct = 1;
    for (int i = 0; i < B_num_rows + 1; i++) {
      if (hB_offsets_tmp[i] != hB_offsets[i]) {
        correct = 0;
        break;
      }
    }
    for (int i = 0; i < B_nnz; i++) {
      if (hB_rows_tmp[i] != hB_rows[i] ||
          fabsf(hB_values_tmp[i] - hB_values[i]) > 1e-2f) {
        correct = 0;                       
        break;
      }
    }

    free(hB);
    free(hB_values);
    free(hB_rows);
    free(hB_offsets);
    free(hB_values_tmp);
    free(hB_rows_tmp);
    free(hB_offsets_tmp);

    if (correct)
      printf("spgeam_example test PASSED\n");
    else
      printf("spgeam_example test FAILED: wrong result\n");
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA ( cudaFree(dA_offsets) )
  CHECK_CUDA ( cudaFree(dA_columns) )
  CHECK_CUDA ( cudaFree(dA_values) )
  CHECK_CUDA ( cudaFree(dB_offsets) )
  CHECK_CUDA ( cudaFree(dB_rows) )
  CHECK_CUDA ( cudaFree(dB_values) )
  free(hA);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}
