/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

/*
#define CUDA_ERRCK \
  { cudaError_t err = cudaGetLastError(); \
    if (err) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
  }
*/

/*
// Place and Transition are implicitly included in the code
// as the grid is a fixed one
typedef struct {
        float mark;
} Place;

typedef struct {
        int from1, from2;
        int to1, to2;
} Transition;

// this starts from row 0 and col 0
P(r,c)    -> T(r,c)   -> P(r,c+1)  ->
  |            |            |
 \/           \/           \/
T(r+1,c-1)-> P(r+1,c) -> T(r+1,c)  ->
  |            |            |
 \/           \/           \/
P(r+2,c)  -> T(r+2,c) -> P(r+2,c+1)->
  |            |            |
 \/           \/           \/
T(r+3,c-1)-> P(r+3,c) -> T(r+3,c)->
  |            |            |
 \/           \/           \/

*/

#include "rand_gen.cu"
#include "petri_kernel.cu"

static int N, S, T, NSQUARE2;
uint32 host_mt[MERS_N];


void* AllocateDeviceMemory(int size);
void CopyFromDeviceMemory(void* h_p, void* d_p, int size);
void CopyFromHostMemory(void* d_p, void* h_p, int size);
void FreeDeviceMemory(void* mem);
void PetrinetOnDevice(long long &ktime);
void compute_statistics();

float results[4];
float* h_vars;
int* h_maxs;

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int main(int argc, char** argv) 
{
  if (argc<4) 
    {
      printf("Usage: %s N S T\n", argv[0]);
      printf("N: the place-transition grid is 2nX2n\n"
             "S: the maximum steps in a trajectory\n"
             "T: number of trajectories\n");
      return -1;
    }

  N = atoi(argv[1]);
  if (N<1)
    return -1;
  S = atoi(argv[2]);
  if (S<1)
    return -1;

  T = atoi(argv[3]);
  if (T<1)
    return -1;

  NSQUARE2 = N*(N+N);
  
  h_vars = (float*)malloc(T*sizeof(float));
  h_maxs = (int*)malloc(T*sizeof(int));
  
  // compute the simulation on the GPU
  long long ktime = 0;

  auto start = get_time();
  
  PetrinetOnDevice(ktime);

  auto end = get_time();

  printf("Total kernel execution time: %.2f s\n", ktime / 1e6f);
  printf("Total device execution time: %.2f s\n", (end - start) / 1e6f);

  compute_statistics();

  free(h_vars);
  free(h_maxs);
    
  printf("petri N=%d S=%d T=%d\n", N, S, T);
  printf("mean_vars: %f    var_vars: %f\n", results[0], results[1]);
  printf("mean_maxs: %f    var_maxs: %f\n", results[2], results[3]);

  return 0;
}

void compute_statistics() 
{
  float sum = 0;
  float sum_vars = 0;
  float sum_max = 0;
  float sum_max_vars = 0;
  int i;
  for (i=0; i<T; i++)
    {
      sum += h_vars[i];
      sum_vars += h_vars[i]*h_vars[i];
      sum_max += h_maxs[i];
      sum_max_vars += h_maxs[i]*h_maxs[i];
    }
  results[0] = sum/T;
  results[1] = sum_vars/T - results[0]*results[0];
  results[2] = sum_max/T;
  results[3] = sum_max_vars/T - results[2]*results[2];
}

void PetrinetOnDevice(long long &time)
{
  // Allocate memory
  int i;
  int unit_size = NSQUARE2*(sizeof(int)+sizeof(char))+
    sizeof(float)+sizeof(int);
  int block_num = MAX_DEVICE_MEM/unit_size;

  printf("Number of thread blocks: %d\n", block_num);

  int *p_hmaxs;
  float *p_hvars;
  int* g_places;
  float* g_vars;
  int* g_maxs;
  
  g_places = (int*)AllocateDeviceMemory((unit_size- sizeof(float) - sizeof(int))*block_num);
  g_vars = (float*)AllocateDeviceMemory(block_num*sizeof(float));
  g_maxs = (int*)AllocateDeviceMemory(block_num*sizeof(int));

  // Setup the execution configuration
  dim3  grid(block_num);  // number of blocks
  dim3  threads(256);  // each block has 256 threads

  p_hmaxs = h_maxs;
  p_hvars = h_vars;

  for (i = 0; i < T-block_num; i += block_num)
  {
    cudaDeviceSynchronize();
    auto start = get_time();

    PetrinetKernel<<<grid, threads>>>
      (g_places, g_vars, g_maxs, N, S, 5489*(i+1));

    cudaDeviceSynchronize();
    auto end = get_time();
    time += end - start;

    CopyFromDeviceMemory(p_hmaxs, g_maxs, block_num*sizeof(int));
    CopyFromDeviceMemory(p_hvars, g_vars, block_num*sizeof(float));

    p_hmaxs += block_num;
    p_hvars += block_num;
  }

  dim3 grid1(T-i);

  cudaDeviceSynchronize();
  auto start = get_time();

  PetrinetKernel<<<grid1, threads>>>
    (g_places, g_vars, g_maxs, N, S, 5489*(i+1));

  cudaDeviceSynchronize();
  auto end = get_time();
  time += end - start;

  // Read result from the device
  CopyFromDeviceMemory(p_hmaxs, g_maxs, (T-i)*sizeof(int));
  CopyFromDeviceMemory(p_hvars, g_vars, (T-i)*sizeof(float));

  // Free device matrices
  FreeDeviceMemory(g_places);
  FreeDeviceMemory(g_vars);
  FreeDeviceMemory(g_maxs);
}

// Allocate a device matrix of same size as M.
void* AllocateDeviceMemory(int size)
{
  int* mem;
  cudaMalloc((void**)&mem, size);
  return mem;
}

// Copy device memory to host memory
void CopyFromDeviceMemory(void* h_p, void* d_p, int size)
{
  cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
  //CUDA_ERRCK
}

// Copy device memory from host memory
void CopyFromHostMemory(void* d_p, void* h_p, int size)
{
  cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
  //CUDA_ERRCK
}

// Free a device matrix.
void FreeDeviceMemory(void* mem)
{
  if (mem!=NULL)
    cudaFree(mem);
  //CUDA_ERRCK
}
