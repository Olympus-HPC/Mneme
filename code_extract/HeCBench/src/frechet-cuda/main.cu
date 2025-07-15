/*
 * Compute the discrete Frechet distance between two curves specified by
 * discrete ordered points in n-dimensional space.
 *
 * Based on `DiscreteFrechetDist` by Zachary Danziger,
 * http://www.mathworks.com/matlabcentral/fileexchange/ \
 * 31922-discrete-frechet-distance
 *
 * This implementation omits the computation of the coupling sequence. Use
 * this program if you only want to get the DFD, and get it fast.
 *
 * Implements algorithm from
 * [1] T. Erepeat and H. Mannila. Computing discrete Frechet distance.
 *     Technical report 94/64, Christian Doppler Laboratory
 *
 * Copyright (c) 2016, Mikhail Pak
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* sqrt, fabs, fmin, fmax */
#include <random>
#include <chrono>
#include <cuda.h>

#define n_d 10000 /* `n_d` : Number of dimensions */

#include "norm1.h"
#include "norm2.h"
#include "norm3.h"

void discrete_frechet_distance(const int s, const int n_1, const int n_2, const int repeat)
{
  double *ca, *c1, *c2;
  int k; /* Index for initialisation of `ca`*/

  int ca_size = n_1*n_2*sizeof(double);
  int c1_size = n_1*n_d*sizeof(double);
  int c2_size = n_2*n_d*sizeof(double);

  /* `ca` : Search array (refer to [1], Table 1, matrix `ca`) */
  ca = (double *) malloc (ca_size);

  /* `c1` and `c2` : Arrays with the 1st and 2nd curve's points respectively */
  c1 = (double *) malloc (c1_size);
  c2 = (double *) malloc (c2_size);

  /* Initialise it with -1.0 */
  for (k = 0; k < n_1*n_2; k++)
  {
    ca[k] = -1.0;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (k = 0; k < n_1 * n_d; k++)
  {
    c1[k] = dis(gen);
  }

  for (k = 0; k < n_2 * n_d; k++)
  {
    c2[k] = dis(gen);
  }

  double *d_ca, *d_c1, *d_c2;
  cudaMalloc((void**)&d_ca, ca_size);
  cudaMalloc((void**)&d_c1, c1_size);
  cudaMalloc((void**)&d_c2, c2_size);

  cudaMemcpy(d_ca, ca, ca_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c1, c1, c1_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c2, c2, c2_size, cudaMemcpyHostToDevice);

  // large n_1 and n_2 values (e.g. n_1 = n_2 = 16) may cause signalTrap
  dim3 grids ((n_1+15)/16, (n_2+15)/16); 
  dim3 blocks (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  if (s == 0)
    for (k = 0; k < repeat; k++)
      distance_norm1<<<grids, blocks>>>(n_1, n_2, d_ca, d_c1, d_c2);

  else if (s == 1)
    for (k = 0; k < repeat; k++)
      distance_norm2<<<grids, blocks>>>(n_1, n_2, d_ca, d_c1, d_c2);

  else if (s == 2)
    for (k = 0; k < repeat; k++)
      distance_norm3<<<grids, blocks>>>(n_1, n_2, d_ca, d_c1, d_c2);
 
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(ca, d_ca, ca_size, cudaMemcpyDeviceToHost);

  double checkSum = 0;
  for (k = 0; k < n_1 * n_2; k++)
    checkSum += ca[k];
  printf("checkSum: %lf\n", checkSum);

  /* Free memory */
  free(ca);
  free(c1);
  free(c2);
  cudaFree(d_ca);
  cudaFree(d_c1);
  cudaFree(d_c2);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <n_1> <n_2> <repeat>\n", argv[0]); 
    printf("  n_1: number of points of the 1st curve");
    printf("  n_2: number of points of the 2nd curve");
    return 1;
  }
                      
  /* `n_1` and `n_2` : Number of points of the 1st and 2nd curves */
  const int n_1 = atoi(argv[1]);
  const int n_2 = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  for (int i = 0; i < 3; i++)
    discrete_frechet_distance(i, n_1, n_2, repeat);

  return 0;
}

