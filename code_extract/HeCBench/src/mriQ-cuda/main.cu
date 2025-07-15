/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <cuda.h>
#include "file.h"
#include "computeQ.cu"

static void
setupMemoryGPU(int num, int size, float*& dev_ptr, float*& host_ptr)
{
  cudaMalloc ((void **) &dev_ptr, num * size);
  CUDA_ERRCK;
  cudaMemcpy (dev_ptr, host_ptr, num * size, cudaMemcpyHostToDevice);
  CUDA_ERRCK;
}

static void
cleanupMemoryGPU(int num, int size, float *& dev_ptr, float * host_ptr)
{
  cudaMemcpy (host_ptr, dev_ptr, num * size, cudaMemcpyDeviceToHost);
  CUDA_ERRCK;
  cudaFree(dev_ptr);
  CUDA_ERRCK;
}

int main (int argc, char *argv[]) {
  char* inputFileName = argv[1];
  char* outputFileName = argv[2];

  int numX, numK;		/* Number of X and K values */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  /* Read in data */
  inputData(inputFileName,
	    &numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  printf("%d pixels in output; %d samples in trajectory\n", numX, numK);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

  /* GPU section 1 (precompute PhiMag) */
  /* Mirror several data structures on the device */
  float *phiR_d, *phiI_d;
  float *phiMag_d;

  setupMemoryGPU(numK, sizeof(float), phiR_d, phiR);
  setupMemoryGPU(numK, sizeof(float), phiI_d, phiI);
  cudaMalloc((void **)&phiMag_d, numK * sizeof(float));
  CUDA_ERRCK;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computePhiMag time: %f s\n", time * 1e-9);

  cleanupMemoryGPU(numK, sizeof(float), phiMag_d, phiMag);
  cudaFree(phiR_d);
  cudaFree(phiI_d);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  float *x_d, *y_d, *z_d;
  float *Qr_d, *Qi_d;

  setupMemoryGPU(numX, sizeof(float), x_d, x);
  setupMemoryGPU(numX, sizeof(float), y_d, y);
  setupMemoryGPU(numX, sizeof(float), z_d, z);
  cudaMalloc((void **)&Qr_d, numX * sizeof(float));
  CUDA_ERRCK;
  cudaMemset((void *)Qr_d, 0, numX * sizeof(float));
  cudaMalloc((void **)&Qi_d, numX * sizeof(float));
  CUDA_ERRCK;
  cudaMemset((void *)Qi_d, 0, numX * sizeof(float));

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  computeQ_GPU(numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computeQ time: %f s\n", time * 1e-9);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cleanupMemoryGPU(numX, sizeof(float), Qr_d, Qr);
  cleanupMemoryGPU(numX, sizeof(float), Qi_d, Qi);
  
  outputData(outputFileName, Qr, Qi, numX);

  free(phiMag);
  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);
  return 0;
}
