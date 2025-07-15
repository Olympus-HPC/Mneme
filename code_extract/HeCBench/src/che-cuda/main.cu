#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>
#include <fstream>
#include <cuda.h>
#include "kernels.h"

using namespace std;

void initialization(double c[][DATAYSIZE][DATAXSIZE])
{
  srand(2);
  for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
    for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
        double f = (double)rand() / RAND_MAX;
        c[idz][idy][idx] = -1.0 + 2.0*f;
      }
    }
  }
}

double integral(const double c[][DATAYSIZE][DATAXSIZE], int nx, int ny, int nz)
{
  double summation = 0.0;  

  for (int k = 0; k < nz; k++)
    for(int j = 0; j < ny; j++)
      for(int i = 0; i < nx; i++)
        summation = summation + c[k][j][i];

  return summation;
}

int main(int argc, char *argv[])
{
  const double dx = 1.0;
  const double dy = 1.0;
  const double dz = 1.0;
  const double dt = 0.01;
  const double e_AA = -(2.0/9.0);
  const double e_BB = -(2.0/9.0);
  const double e_AB = (2.0/9.0);
  const int t_f = atoi(argv[1]);    // default value: 25000
#ifndef DEBUG
  const int t_freq = t_f; 
#else
  const int t_freq = 10;
#endif
  const double gamma = 0.5;
  const double D = 1.0;

  string name_c = "./out/integral_c.txt";
  ofstream ofile_c (name_c);

  string name_mu = "./out/integral_mu.txt";
  ofstream ofile_mu (name_mu);

  string name_f = "./out/integral_f.txt";
  ofstream ofile_f (name_f);

  typedef double nRarray[DATAYSIZE][DATAXSIZE];

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_bytes = vol * sizeof(double);

  // pointers for data set storage via malloc
  nRarray *c_host; // storage for result stored on host
  nRarray *mu_host;
  nRarray *f_host;
  nRarray *d_cold; // storage for result computed on device
  nRarray *d_cnew;
  nRarray *d_muold;
  nRarray *d_fold;

  if ((c_host = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"c_host malloc failed\n"); 
    return 1;
  }
  if ((mu_host = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"mu_host malloc failed\n"); 
    return 1;
  }
  if ((f_host = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"f_host malloc failed\n"); 
    return 1;
  }

  cudaMalloc((void **) &d_cold, vol_bytes);
  cudaMalloc((void **) &d_cnew, vol_bytes);
  cudaMalloc((void **) &d_muold, vol_bytes);
  cudaMalloc((void **) &d_fold, vol_bytes);

  initialization(c_host);

  double integral_c = 0.0;
  double integral_mu = 0.0;
  double integral_f = 0.0;

  cudaMemcpy(d_cold, c_host, vol_bytes, cudaMemcpyHostToDevice);

  const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
  const dim3 gridSize((DATAXSIZE+BLKXSIZE-1)/BLKXSIZE, 
                      (DATAYSIZE+BLKYSIZE-1)/BLKYSIZE,
                      (DATAZSIZE+BLKZSIZE-1)/BLKZSIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int t = 0; t < t_f; t++) {

    chemicalPotential<<<gridSize, blockSize>>>(d_cold,d_muold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    localFreeEnergyFunctional<<<gridSize, blockSize>>>(d_cold,d_fold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    cahnHilliard<<<gridSize, blockSize>>>(d_cnew,d_cold,d_muold,D,dt,dx,dy,dz);

    if (t > 0 && t % (t_freq - 1) == 0) {
      cudaMemcpy(c_host, d_cnew, vol_bytes, cudaMemcpyDeviceToHost);

      cudaMemcpy(mu_host, d_muold, vol_bytes, cudaMemcpyDeviceToHost);

      cudaMemcpy(f_host, d_fold, vol_bytes, cudaMemcpyDeviceToHost);

      integral_c = integral(c_host,nx,ny,nz);

      ofile_c << t << "," << integral_c << endl;

      integral_mu = integral(mu_host,nx,ny,nz);

      ofile_mu << t << "," << integral_mu << endl;

      integral_f = integral(f_host,nx,ny,nz);

      ofile_f << t << "," << integral_f << endl;
    }

    Swap<<<gridSize, blockSize>>>(d_cnew, d_cold);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel exeuction time on the GPU (%d iterations) = %.3f (s)\n", t_f, time * 1e-9f);

  free(c_host);
  free(mu_host);
  free(f_host);
  cudaFree(d_cold);
  cudaFree(d_cnew);
  cudaFree(d_muold);
  cudaFree(d_fold);
  return 0;
}
