/*---------------------------------------------------------------
  Original author: Zebulun Arendsee
  March 26, 2013
----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

#define PI 3.14159265359f
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ADD 256

/* 
   Box-Muller Transformation: Generate one standard normal variable.

   This algorithm can be more efficiently used by producing two
   random normal variables. However, for the CPU, much faster
   algorithms are possible (e.g. the Ziggurat Algorithm);

   This is actually the algorithm chosen by nVidia to calculate
   normal random variables on the GPU.
 */
__host__ float rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  float V1 = sqrtf(-2 * logf(U1)) * cosf(2 * PI * U2);
  return V1;
}

/*
   Generate random gamma variables on a CPU.
 */
__host__ float rgamma(float a, float b)
{
  float x;
  if (a > 1) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    bool flag = true;
    float V;
    while (flag) {
      // Generate a standard normal random variable
      float Z = rnorm();
      if (Z > -1.f / c) {
        V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
      }
    }
    x = d * V / b;
  }
  else 
  {
    x = rgamma(a + 1.f, b);
    x = x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
  return x;
}

/*
   Metropolis algorithm for producing random a values. 
   The proposal distribution in normal with a variance that
   is adjusted at each step.
 */
__host__ float sample_a(float a, float b, int N, float log_sum)
{
  static float sigma = 2;

  float proposal = rnorm() * sigma + a;

  if(proposal <= 0) return a;

  float log_acceptance_ratio = (proposal - a) * log_sum +
    N * (proposal - a) * logf(b) -
    N * (lgamma(proposal) - lgamma(a));

  float U = rand() / float(RAND_MAX);

  if(logf(U) < log_acceptance_ratio){
    sigma *= 1.1;
    return proposal;
  } else {
    sigma /= 1.1;
    return a;
  }
}

/*
   Returns a random b sample (simply a draw from the appropriate
   gamma distribution)
 */
__host__ float sample_b(float a, int N, float flat_sum)
{
  float hyperA = N * a + 1;
  float hyperB = flat_sum;
  return rgamma(hyperA, hyperB);
}

/* 
   Generate a single Gamma distributed random variable by the Marsoglia 
   algorithm (George Marsaglia, Wai Wan Tsang; 2001).
 */
__device__ float rgamma(curandState *state, float a, float b)
{
  float d = a - 1.f / 3.f;
  float c = 1.f / sqrtf(9.f * d);
  bool flag = true;
  float V;

  while (flag) {
    // Generate a standard normal random variable
    float Z = curand_normal(state);
    if (Z > -1.f / c) {
      V = powf(1.f + c * Z, 3.f);
      float U = curand_uniform(state);
      flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
    }
  }
  return d * V / b;
}

/* 
   Initializes GPU random number generators 
 */
__global__ void setup_kernel(curandState *state, int num_sample, unsigned int seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < num_sample)
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

/*
   Sample each theta from the appropriate gamma distribution
 */
__global__ void sample_theta(
    curandState *__restrict__ state, 
    float *__restrict__ theta,
    const int *__restrict__ y,
    const float *__restrict__ n, 
    float a, float b, int num_sample)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < num_sample) {
    curandState *s = state + id;
    const float hyperA = a + y[id];
    const float hyperB = b + n[id];
    theta[id] = (hyperA < 1.f) ?
      rgamma(s, hyperA + 1.f, hyperB) * powf(curand_uniform(s), 1.f / hyperA) : 
      rgamma(s, hyperA, hyperB);
  }
}

/* 
   Sampling of a and b require the sum and product of all theta 
   values. This function performs parallel summations of 
   flat values and logs of theta for many blocks of length
   THREADS_PER_BLOCK_ADD. The CPU will then sum the block
   sums (see main method);    
 */
__global__ void sum_blocks(const float *__restrict__ theta,
                                 float *__restrict__ flat_sums, 
                                 float *__restrict__ log_sums,
                                   int num_sample)
{
  __shared__ float flats[THREADS_PER_BLOCK_ADD];
  __shared__ float logs[THREADS_PER_BLOCK_ADD];

  int id = threadIdx.x + blockIdx.x * blockDim.x;

  flats[threadIdx.x] = (id < num_sample) ? theta[id] : 0.f;
  logs[threadIdx.x] = (id < num_sample) ? logf( theta[id] ) : 0.f;

  __syncthreads();

  int i = blockDim.x / 2;
  while(i != 0){
    if(threadIdx.x < i){
      flats[threadIdx.x] += flats[threadIdx.x + i];
      logs[threadIdx.x] += logs[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    flat_sums[blockIdx.x] = flats[0];
    log_sums[blockIdx.x] = logs[0];
  }
}

int main(int argc, char *argv[])
{
  const int seed = 123; // seed for random number generation
  srand(seed);

  int trials = 1000; // Default number of trials
  if(argc > 2) {
    trials = atoi(argv[2]);
  }

  /*------ Loading Data ------------------------------------------*/

  FILE *fp;
  if(argc > 1){
    fp = fopen(argv[1], "r");
  } else {
    printf("Please provide input filename\n");
    return 1;
  }

  if(fp == NULL){
    printf("Cannot read file \n");
    return 1;
  }

  int N = 0;
  char line[128];
  while( fgets (line, sizeof line, fp) != NULL ) { 
    N++; 
  }

  rewind(fp);

  size_t y_size = sizeof(int) * N;
  size_t n_size = sizeof(float) * N;
  int *y = (int*) malloc (y_size);
  float *n = (float*) malloc (n_size);
  float *host_fpsum = (float*) malloc (n_size);
  float *host_lpsum = (float*) malloc (n_size);

  for(int i = 0; i < N; i++) {
    fscanf(fp, "%d %f", &y[i], &n[i]);    
  }

  fclose(fp);

  /*------ Memory Allocations ------------------------------------*/
  float *dev_theta, *dev_fpsum, *dev_lpsum, *dev_n;
  int *dev_y;

  CUDA_CALL(cudaMalloc((void **)&dev_y, y_size));
  CUDA_CALL(cudaMemcpy(dev_y, y, y_size, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc((void **)&dev_n, n_size));
  CUDA_CALL(cudaMemcpy(dev_n, n, n_size, cudaMemcpyHostToDevice));

  // Allocate memory for the partial flat and log sums
  int nSumBlocks = (N + (THREADS_PER_BLOCK_ADD - 1)) / THREADS_PER_BLOCK_ADD;
  CUDA_CALL(cudaMalloc((void **)&dev_fpsum, nSumBlocks * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&dev_lpsum, nSumBlocks * sizeof(float)));

  /* Allocate space for theta on device and host */
  CUDA_CALL(cudaMalloc((void **)&dev_theta, n_size));

  /* Allocate space for random states on device */
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **)&devStates, N * sizeof(curandState)));

  /*------ Setup RNG ---------------------------------------------*/

  int nBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Setup the rng machines (one for each thread)
  setup_kernel<<<nBlocks, THREADS_PER_BLOCK>>>(devStates, N, seed);

  /*------ MCMC Algorithm ----------------------------------------*/
  float a = 20; // Set starting value
  float b = 1;  // Set starting value

  double mean_a = 0.0, mean_b = 0.0;
  double total_time = 0.0;

  for(int i = 0; i < trials; i++) {
    auto start = std::chrono::steady_clock::now();

    sample_theta<<<nBlocks, THREADS_PER_BLOCK>>>(
      devStates, dev_theta, dev_y, dev_n, a, b, N);

    // Sum the flat and log values of theta 
    sum_blocks<<<nSumBlocks, THREADS_PER_BLOCK_ADD>>>(
      dev_theta, dev_fpsum, dev_lpsum, N);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time; 

    CUDA_CALL(cudaMemcpy(host_fpsum, dev_fpsum, nSumBlocks * sizeof(float), 
          cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaMemcpy(host_lpsum, dev_lpsum, nSumBlocks * sizeof(float), 
          cudaMemcpyDeviceToHost));

    // The GPU summed blocks of theta values, now the CPU sums the blocks
    float flat_sum = 0;
    float log_sum = 0; 
    for(int j = 0; j < nSumBlocks; j++) {
      flat_sum += host_fpsum[j];
      log_sum += host_lpsum[j];
    }

    // Sample one random value from a's distribution
    a = sample_a(a, b, N, log_sum);
    mean_a += a;

    // And then from b's distribution given the new a
    b = sample_b(a, N, flat_sum);
    mean_b += b;
  }

  printf("Average execution time of kernels: %f (us)\n", (total_time * 1e-3f) / trials);
  printf("a = %lf (avg), b = %lf (avg)\n", mean_a / trials, mean_b / trials);

  /*------ Free Memory -------------------------------------------*/

  CUDA_CALL(cudaFree(devStates));
  CUDA_CALL(cudaFree(dev_theta));
  CUDA_CALL(cudaFree(dev_fpsum));
  CUDA_CALL(cudaFree(dev_lpsum));
  CUDA_CALL(cudaFree(dev_y));
  CUDA_CALL(cudaFree(dev_n));
  free(host_fpsum);
  free(host_lpsum);
  free(y);
  free(n);

  return EXIT_SUCCESS;
}
