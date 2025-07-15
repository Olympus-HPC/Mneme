/*
 * Copyright 2014      ARM Ltd.
 *
 * Use of this software is governed by the MIT license
 *
 * Compared to the original C example, the cpu and gpu results 
 * are compared for verification. -Zheming Jin
 *   
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "kernel.h"

#define REPEAT 1000
#define N 370
#define LDAT N
#define INCX 1
#define INCY 1
#define AT_SIZE (N * LDAT)
#define X_SIZE (N * INCX)
#define Y_SIZE (N * INCY)


void chemv_cpu(float alpha_re, float alpha_im, float beta_re, float beta_im,
               struct ComplexFloat AT[AT_SIZE], struct ComplexFloat X[X_SIZE],
               struct ComplexFloat Y[Y_SIZE])
{
  for (int i0 = 0; i0 <= (N - 1); i0 += 1) {
    float var5_Re;
    float var5_Im;
    var5_Re =
      ((Y[i0 * INCY + 0].Re * beta_re) - (Y[i0 * INCY + 0].Im * beta_im));
    var5_Im =
      ((Y[i0 * INCY + 0].Im * beta_re) + (Y[i0 * INCY + 0].Re * beta_im));
    Y[i0 * INCY + 0].Re = var5_Re;
    Y[i0 * INCY + 0].Im = var5_Im;
  }

  for (int i1 = 0; i1 <= ((N - 1) + 1) - 1; i1 += 1) {
    float var2_Re;
    float var3_Im;
    float var2_Im;
    float var4_Im;
    float var4_Re;
    float var3_Re;
    var2_Re = (alpha_re * AT[i1 * LDAT + i1].Re);
    var2_Im = (alpha_im * AT[i1 * LDAT + i1].Re);
    var3_Re = ((var2_Re * X[i1 * INCX + 0].Re) - (var2_Im * X[i1 * INCX + 0].Im));
    var3_Im = ((var2_Im * X[i1 * INCX + 0].Re) + (var2_Re * X[i1 * INCX + 0].Im));
    var4_Re = (Y[i1 * INCY + 0].Re + var3_Re);
    var4_Im = (Y[i1 * INCY + 0].Im + var3_Im);
    Y[i1 * INCY + 0].Re = var4_Re;
    Y[i1 * INCY + 0].Im = var4_Im;
  }

  for (int i2 = 0; i2 <= ((N - 1) - 1); i2 += 1) {
    for (int i3 = 0; i3 <= (N - 1) - (1 + i2); i3 += 1) {
      float var99_Re;
      float var96_Re;
      float var98_Im;
      float var96_Im;
      float var94_Im;
      float var95_Im;
      float var94_Re;
      float var95_Re;
      float var97_Im;
      float var99_Im;
      float var97_Re;
      float var98_Re;
      var94_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      var94_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      var95_Re = ((var94_Re * X[((i3 + i2) + 1) * INCX + 0].Re) -
          (var94_Im * X[((i3 + i2) + 1) * INCX + 0].Im));
      var95_Im = ((var94_Im * X[((i3 + i2) + 1) * INCX + 0].Re) +
          (var94_Re * X[((i3 + i2) + 1) * INCX + 0].Im));
      var96_Re = (Y[i2 * INCY + 0].Re + var95_Re);
      var96_Im = (Y[i2 * INCY + 0].Im + var95_Im);
      Y[i2 * INCY + 0].Re = var96_Re;
      Y[i2 * INCY + 0].Im = var96_Im;
      var97_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      var97_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      var98_Re = ((var97_Re * X[i2 * INCX + 0].Re) -
          (var97_Im * X[i2 * INCX + 0].Im));
      var98_Im = ((var97_Im * X[i2 * INCX + 0].Re) +
          (var97_Re * X[i2 * INCX + 0].Im));
      var99_Re = (Y[((i3 + i2) + 1) * INCY + 0].Re + var98_Re);
      var99_Im = (Y[((i3 + i2) + 1) * INCY + 0].Im + var98_Im);
      Y[((i3 + i2) + 1) * INCY + 0].Re = var99_Re;
      Y[((i3 + i2) + 1) * INCY + 0].Im = var99_Im;
    }
  }
}

/* chemv - complex hermitian matrix-vector multiplication
 * The function body was taken from a VOBLA-generated BLAS library.
 */
void chemv_gpu(float alpha_re, float alpha_im, float beta_re, float beta_im,
               struct ComplexFloat AT[AT_SIZE], struct ComplexFloat X[X_SIZE],
               struct ComplexFloat Y[Y_SIZE])
{
  struct ComplexFloat *dev_AT;
  struct ComplexFloat *dev_X;
  struct ComplexFloat *dev_Y;

  cudaMalloc((void **) &dev_AT, AT_SIZE * sizeof(struct ComplexFloat));
  cudaMalloc((void **) &dev_X, X_SIZE * sizeof(struct ComplexFloat));
  cudaMalloc((void **) &dev_Y, Y_SIZE * sizeof(struct ComplexFloat));

  cudaMemcpy(dev_AT, AT, AT_SIZE * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_X, X, X_SIZE * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Y, Y, Y_SIZE * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice);

  dim3 k0_dimBlock(32);
  dim3 k0_dimGrid(12);

  dim3 k1_dimBlock(32);
  dim3 k1_dimGrid(12);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < REPEAT; n++) {
    kernel0 <<< k0_dimGrid, k0_dimBlock >>> (dev_AT, dev_X, dev_Y, alpha_im, alpha_re, beta_im, beta_re);
    kernel1 <<< k1_dimGrid, k1_dimBlock >>> (dev_AT, dev_X, dev_Y, alpha_im, alpha_re);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of chemv kernels: %f (us)\n", (time * 1e-3f) / REPEAT);

  cudaMemcpy(Y, dev_Y, Y_SIZE * sizeof(struct ComplexFloat), cudaMemcpyDeviceToHost);
  cudaFree(dev_AT);
  cudaFree(dev_X);
  cudaFree(dev_Y);
}

int main() {
  struct ComplexFloat AT[AT_SIZE];
  struct ComplexFloat X[X_SIZE];
  struct ComplexFloat Y_cpu[Y_SIZE];
  struct ComplexFloat Y_gpu[Y_SIZE];

  for (int i = 0; i < N; i++) {
    X[i * INCX + 0] = (struct ComplexFloat){static_cast<float>(i + 5), static_cast<float>(i * 2)};
    Y_cpu[i * INCY + 0] = (struct ComplexFloat){static_cast<float>(i * 3), static_cast<float>(i + 7)};
    Y_gpu[i * INCY + 0] = (struct ComplexFloat){static_cast<float>(i * 3), static_cast<float>(i + 7)};
    for (int j = 0; j < LDAT; j++) {
      AT[i * LDAT + j] = (struct ComplexFloat){static_cast<float>(i + j), static_cast<float>(i + 3)};
    }
  }

  const float alpha_re = 3.14f;
  const float alpha_im = 1.59f;
  const float beta_re = 2.71f;
  const float beta_im = 8.28f;

  chemv_cpu(alpha_re, alpha_im, beta_re, beta_im, AT, X, Y_cpu);
  chemv_gpu(alpha_re, alpha_im, beta_re, beta_im, AT, X, Y_gpu);

  for (int i = 0; i < N; i++)
    if ((fabs(Y_cpu[i * INCY + 0].Re - Y_gpu[i * INCY + 0].Re) > 1e-3) ||
        (fabs(Y_cpu[i * INCY + 0].Im - Y_gpu[i * INCY + 0].Im) > 1e-3))
    {
      printf("%d %f %f\n", i, Y_cpu[i * INCY + 0].Re,Y_gpu[i * INCY + 0].Re);
      printf("FAILED\n");
      return EXIT_FAILURE;
    }
  printf("PASSED\n");
  return EXIT_SUCCESS;
}

