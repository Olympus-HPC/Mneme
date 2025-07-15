/*******************************************************************************
 * Copyright 2022 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted  materials, and
 * your use of  them is  governed by the  express license  under which  they
 *were provided to you (License).  Unless the License provides otherwise, you
 *may not use, modify, copy, publish, distribute,  disclose or transmit this
 *software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents  are provided as  is,  with no
 *express or implied  warranties,  other  than those  that are  expressly stated
 *in the License.
 *******************************************************************************/

/*
 *  Content:
 *       This example demonstrates use of cublasXgelsBatched
 *       to perform batched calculation of least squares.
 *
 *       The supported floating point data types for matrix data are:
 *           float
 *           double
 *           std::complex<float>
 *           std::complex<double>
 *******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <iostream>
#include <cublas_v2.h>
#include <cuda.h>

template <typename data_t, typename real_t = decltype(std::real((data_t)0)),
          bool is_real = std::is_same_v<data_t, real_t>>
int run_gels_batch_example(const int repeat) {
  
  const int64_t m = 5, n = 5, nrhs = 1, lda = m, stride_a = n * lda, ldb = m,
                stride_b = nrhs * ldb, batch_size = 2;

  auto v = [](real_t arg) {
    if constexpr (is_real)
      return arg;
    else
      return data_t{0, arg};
  };

  data_t A[] = {
      v( 1.0), v( 0.0), v( 0.0), v( 0.0), v( 0.0),
      v( 1.0), v( 0.2), v(-0.4), v(-0.4), v(-0.8),
      v( 1.0), v( 0.6), v(-0.2), v( 0.4), v(-1.2),
      v( 1.0), v( 1.0), v(-1.0), v( 0.6), v(-0.8),
      v( 1.0), v( 1.8), v(-0.6), v( 0.2), v(-0.6)
                                                 ,
      v( 0.2), v(-0.4), v(-0.4), v(-0.8), v( 0.0),
      v( 0.4), v( 0.2), v( 0.8), v(-0.4), v( 0.0),
      v( 0.4), v(-0.8), v( 0.2), v( 0.4), v( 0.0),
      v( 0.8), v( 0.4), v(-0.4), v( 0.2), v( 0.0),
      v( 0.0), v( 0.0), v( 0.0), v( 0.0), v( 1.0)
  };

  data_t B[] = {
      v(5.0), v(3.6),  v(-2.2), v(0.8),  v(-3.4),
      v(1.8), v(-0.6), v(0.2),  v(-0.6), v(1.0),
  };

  data_t X[] = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };

  cublasHandle_t h;
  cublasStatus_t status;
  status = cublasCreate(&h);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS initialization failed..\n");
    return (EXIT_FAILURE);
  }

  data_t *A_dev;
  cudaMalloc((void**)&A_dev, stride_a * batch_size * sizeof(data_t));

  data_t *B_dev;
  cudaMalloc((void**)&B_dev, stride_b * batch_size * sizeof(data_t));

  data_t** ptrA_array = (data_t**) malloc (batch_size * sizeof(data_t*));
  for (int i = 0; i < batch_size; i++) ptrA_array[i] = A_dev + (i * stride_a);

  data_t** ptrB_array = (data_t**) malloc (batch_size * sizeof(data_t*));
  for (int i = 0; i < batch_size; i++) ptrB_array[i] = B_dev + (i * stride_b);

  data_t **ptrA_array_dev, **ptrB_array_dev;
  cudaMalloc((void**)&ptrA_array_dev, batch_size * sizeof(data_t*));
  cudaMalloc((void**)&ptrB_array_dev, batch_size * sizeof(data_t*));
  cudaMemcpy(ptrA_array_dev, ptrA_array, batch_size * sizeof(data_t*), cudaMemcpyHostToDevice);
  cudaMemcpy(ptrB_array_dev, ptrB_array, batch_size * sizeof(data_t*), cudaMemcpyHostToDevice);
  int info;

  long time = 0;
  for (int i = 0; i <= repeat; i++) {
    cudaMemcpy(A_dev, A, stride_a * batch_size * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, stride_b * batch_size * sizeof(data_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    if constexpr (std::is_same_v<data_t, float>)
      status = cublasSgelsBatched(h, CUBLAS_OP_N, m, n, nrhs, ptrA_array_dev, lda,
                                  ptrB_array_dev, ldb, &info, NULL, batch_size);
    else if constexpr (std::is_same_v<data_t, double>)
      status = cublasDgelsBatched(h, CUBLAS_OP_N, m, n, nrhs, ptrA_array_dev, lda,
                                  ptrB_array_dev, ldb, &info, NULL, batch_size);
    else if constexpr (std::is_same_v<data_t, std::complex<float>>)
      status = cublasCgelsBatched(h, CUBLAS_OP_N, m, n, nrhs,
                                  reinterpret_cast<cuComplex *const *>(ptrA_array_dev),
                                  lda,
                                  reinterpret_cast<cuComplex *const *>(ptrB_array_dev),
                                  ldb, &info, NULL, batch_size);
    else if constexpr (std::is_same_v<data_t, std::complex<double>>)
      status = cublasZgelsBatched(h, CUBLAS_OP_N, m, n, nrhs,
                                  reinterpret_cast<cuDoubleComplex *const *>(ptrA_array_dev),
                                  lda,
                                  reinterpret_cast<cuDoubleComplex *const *>(ptrB_array_dev),
                                  ldb, &info, NULL, batch_size);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    if (i != 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("> ERROR: cublasXgelsBatched() failed with error %s..\n",
             cublasGetStatusString(status));
    }
    // If info=0, the parameters passed to the function are valid
    // If info<0, the parameter in position -info is invalid
    if (info < 0)
     printf("The parameter in position %d is invalid\n", -info);
  }
  printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(B, B_dev, stride_b * batch_size * sizeof(data_t), cudaMemcpyDeviceToHost);

  const real_t bound = std::is_same_v<real_t, float> ? 1e-6 : 1e-8;
  bool passed = true;

  printf("Results:\n");
  auto print = [](data_t &v) {
    if constexpr (is_real)
      printf("%6.2f", v);
    else
      printf("<%6.2f,%6.2f> ", v.real(), v.imag());
  };
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < n; j++) {
      data_t result = B[i * stride_b + j];
      data_t residual = result - X[j + i * m];
      passed = passed and (result == result) and
               (std::sqrt(std::abs(std::real(residual * residual))) < bound);
      print(result);
    }
    printf("\n");
  }

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(ptrA_array_dev);
  cudaFree(ptrB_array_dev);
  status = cublasDestroy(h);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS uninitialization failed..\n");
  }

  free(ptrA_array);
  free(ptrB_array);

  if (passed) {
    printf("Calculations successfully finished\n");
  } else {
    printf("ERROR: results mismatch!\n");
    printf("Expected:\n");
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < m; j++) {
        print(X[j + i * m]);
      }
      printf("\n");
    }
    return 1;
  }

  return 0;
}

//
// Description of example setup, APIs used and supported floating point type
// precisions
//
void print_info() {
  std::cout << "" << std::endl;
  std::cout << "########################################################################" << std::endl;
  std::cout << "# Batched strided GELS example:" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "# Computes least squares of a batch of matrices and right hand sides." << std::endl;
  std::cout << "# Supported floating point type precisions:" << std::endl;
  std::cout << "#   float" << std::endl;
  std::cout << "#   double" << std::endl;
  std::cout << "#   std::complex<float>" << std::endl;
  std::cout << "#   std::complex<double>" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "########################################################################" << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  print_info();

  bool failed = false;

  std::cout << "Running with single precision real data type:" << std::endl;
  failed |= run_gels_batch_example<float>(repeat);

  std::cout << "Running with single precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<std::complex<float>>(repeat);

  std::cout << "Running with double precision real data type:" << std::endl;
  failed |= run_gels_batch_example<double>(repeat);

  std::cout << "Running with double precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<std::complex<double>>(repeat);

  return failed;
}
