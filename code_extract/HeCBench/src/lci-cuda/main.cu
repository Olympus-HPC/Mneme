#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "tables.h"
#include "kernels.h"

const double t_init = .1;
const double t_final = 200;

void initial(double c[], int seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dis(-6.0, 6.0);
  std::uniform_real_distribution<double> temp(0.1, 0.9);
  for(int l = 1; l < L_max; l++) c[l] = dis(gen);
  c[0] = temp(gen) ; // random temperature
  c[L_max] = 0.0; //truncation; do not modify
}

#ifdef DUMP
void dump (FILE * out, double c[], double t)
{
  fprintf(out,"%.5e ", t);
  int L = (L_max > 4) ? 4 : L_max; // print a subset 
  for(int l = 0; l < L; l++)
  {
    fprintf(out,"%.5e ", c[l]);
  }
  fprintf(out,"\n");
  fflush(out);
}
#endif

int main (int argc, char* argv[]) {

#ifdef DUMP
  FILE * outdata = fopen ("output.csv","w");
  if (outdata == NULL) {
    printf("Failed to open file for write\n");
    return 1;
  }
#endif

  double tmin = t_init;
  double tmax = t_final;
  double delta_t = 0.1;

  int dimension = L_max;
  int seed = dimension;

  int size = sizeof(double) * (dimension+1);

  double *c = (double*) malloc (size);
  double *n = (double*) malloc (size);

  initial(c, seed);
#ifdef DUMP
  dump(outdata, c, tmin);
#endif

  double *d_c, *d_n;
  cudaMalloc((void**)&d_c, size);
  cudaMalloc((void**)&d_n, size);

  float total_time = 0.f;
  for (double t_next = tmin + delta_t; t_next <= tmax; t_next += delta_t)
  {
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice); 

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    RHS_f <<<1, 96>>> (t_next, d_c, d_n);  // work-group size >= L_max

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

#ifdef DUMP
    cudaMemcpy(n, d_n, size, cudaMemcpyDeviceToHost); 
    dump(outdata, n, t_next);
#endif
    initial(c, ++seed);
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);

#ifdef DUMP
  fclose (outdata);
#endif

  cudaFree(d_c);
  cudaFree(d_n);
  free(c);
  free(n);

  return 0;
}
