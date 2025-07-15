#include <sys/time.h>
#include <stdio.h>
#include <type_traits> // is_same
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include "utils.h"

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: status: %d, line: %d in %s\n", status, line, file);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;

cublasLtHandle_t handle;

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
  cudaMallocManaged(A, m * k * sizeof(T));
  cudaMallocManaged(B, k * n * sizeof(T));
  cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

template <typename T, typename S, typename CT>
bool cublasLt_gemm(
    const int m, const int n, const int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    const CT *alpha, const CT *beta, int compute32F_mode)
{
  bool status = true;
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatmulPreference_t preference;
  cublasLtMatrixLayout_t ALayout;
  cublasLtMatrixLayout_t BLayout;
  cublasLtMatrixLayout_t CLayout;
  cublasLtMatmulHeuristicResult_t heuristic;

  // create the operation descriptor
  //cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasOperation_t opNoTranspose = CUBLAS_OP_N;

  cudaDataType_t AType, CType, SType;
  cublasComputeType_t ComputeType;
  if (std::is_same<T, double>::value) {
    AType = SType = CType = CUDA_R_64F;
    ComputeType = CUBLAS_COMPUTE_64F;
  } else if (std::is_same<T, float>::value) {
    AType = SType = CType = CUDA_R_32F;
    switch (compute32F_mode) {
      case 0: ComputeType = CUBLAS_COMPUTE_32F;
      case 1: ComputeType = CUBLAS_COMPUTE_32F_FAST_16F;
      case 2: ComputeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      case 3: ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      default: ComputeType = CUBLAS_COMPUTE_32F;
    }
  } else if (std::is_same<T, __half>::value) {
    AType = CType = CUDA_R_16F;
    if (std::is_same<CT, __half>::value) {
      SType = CUDA_R_16F;
      ComputeType = CUBLAS_COMPUTE_16F;
    }
    else {
      SType = CUDA_R_32F;
      ComputeType = CUBLAS_COMPUTE_32F;
    }
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    AType = CType = CUDA_R_16BF;
    SType = CUDA_R_32F;
    ComputeType = CUBLAS_COMPUTE_32F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = CUDA_R_8I;
    CType = SType = CUDA_R_32I;
    ComputeType = CUBLAS_COMPUTE_32I;
  } else {
    printf("Not supported data type.");
    return false;
  }
  
  // compute type and scale type
  cublasCheck(cublasLtMatmulDescCreate(&operationDesc, ComputeType, SType));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
              CUBLASLT_MATMUL_DESC_TRANSA, &opNoTranspose, sizeof(opNoTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(
              operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));

  // define matrix layouts
  cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, AType, m, k, lda));
  cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, AType, k, n, ldb));
  cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CType, m, n, ldc));

  // create a preference handle with specified max workspace
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // find a suitable algorithm
  int returnedResults = 0;
  cublasCheck(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc,
      ALayout, BLayout, CLayout, CLayout,
      preference, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
      printf("No cuBLASLt algorithm found.\n");
      status = false;
      goto cleanups;
  }

  // call the matmul
  cublasCheck(cublasLtMatmul(handle, operationDesc,
      alpha, B, ALayout, A, BLayout, beta,
      C, CLayout, C, CLayout, &heuristic.algo,
      cublaslt_workspace, cublaslt_workspace_size, 0));

  // wait for matmul
  cudaDeviceSynchronize();

  cleanups:
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));

  return status;
}

template <typename T, typename S, typename CT>
void test_gemm(//cublasLtHandle_t handle,
  const int m,  const int n,  const int k,
  T *A, T *B, S *C,
  const CT *alpha, const CT *beta, const int iteration,
  int compute32F_mode = 0)
{
  double total_time = 0;
  struct timeval start, end;

  for (int i = 0; i < iteration; ++i) {
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    bool success = cublasLt_gemm( n, // number of rows of matrix A and C
                                  m, // number of columns of matrix B and C
                                  k, // number of columns of A and rows of B
                                  B,
                                  A,
                                  C,
                                  n, // lda
                                  k, // ldb
                                  n, // ldc
                                  alpha,
                                  beta,
                                  compute32F_mode);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    if (!success) break;
    else if (i > 0) {
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    performance(m, n, k, std::is_same<T, int8_t>::value, avg_time * 1e-3);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  const __half h_alpha = __float2half_rn(1.f),
               h_beta = __float2half_rn(0.f);
  const float h_alpha2 = 1.f, h_beta2 = 0.f;
  const float bf_alpha(1.f), bf_beta(0.f);
  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  __nv_bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(m, n, k, &dA, &dB, &dC);
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    bfA[i] = __float2bfloat16_rn(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    bfB[i] = __float2bfloat16_rn(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }

  cublasCheck(cublasLtCreate(&handle));
  cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, dA, dB, dC, &d_alpha, &d_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type tf32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration, 3);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type bf16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration, 2);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type fp16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration, 1);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, hA, hB, hC, &h_alpha, &h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, hA, hB, hC, &h_alpha2, &h_beta2, iteration);

  printf(">>>>>>>>>>>>>>>>> test bfloat16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, bfA, bfB, bfC, &bf_alpha, &bf_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, iA, iB, iC, &i_alpha, &i_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", dC[i], " \n"[i==9]);

  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

  printf("bf16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(bfC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

  cudaFree(cublaslt_workspace);
  cublasCheck(cublasLtDestroy(handle));
  free_memory(dA, dB, dC);
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  free_memory(bfA, bfB, bfC);
  free_memory(iA, iB, iC);
  return 0;
}
