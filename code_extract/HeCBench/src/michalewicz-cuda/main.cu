#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>

__device__ __forceinline__
float atomic_min(float *addr, float value)
{
  unsigned ret = __float_as_uint(*addr);
  while(value < __uint_as_float(ret))
  {
    unsigned old = ret;
    if((ret = atomicCAS((unsigned *)addr, old, __float_as_uint(value))) == old)
      break;
  }
  return __uint_as_float(ret);
}

__device__ __forceinline__
float michalewicz(const float *xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
    float a = sinf(xValues[i]);
    float b = sinf(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
    float c = powf(b, 20); // m = 10
    result += a * c;
  }
  return -1.0f * result;
}

__global__ void eval (const float *values, float *minima,
                      const size_t nVectors, const int dim)
{
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < nVectors) {
    atomic_min(minima, michalewicz(values + n * dim, dim));
  }
}

// https://www.sfu.ca/~ssurjano/michal.html
void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0;
  if (dim == 2)
    trueMin = -1.8013;
  else if (dim == 5)
    trueMin = -4.687658;
  else if (dim == 10)
    trueMin = -9.66015;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  // generate random numbers
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0, 4.0);
  
  // dimensions
  const int dims[] = {2, 5, 10}; 

  for (int d = 0; d < 3; d++) {

    const int dim = dims[d];

    const size_t size = n * dim;

    const size_t size_bytes = size * sizeof(float);
    
    float *values = (float*) malloc (size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }
    
    float *d_values;
    cudaMalloc((void**)&d_values, size_bytes);
    cudaMemcpy(d_values, values, size_bytes, cudaMemcpyHostToDevice);

    float *d_minValue;
    float minValue;
    cudaMalloc((void**)&d_minValue, sizeof(float));

    dim3 grids ((n + 255) / 256);
    dim3 blocks (256);

    cudaMemset(d_minValue, 0, sizeof(float));
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      eval<<<grids, blocks>>>(d_values, d_minValue, n, dim);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    cudaMemcpy(&minValue, d_minValue, sizeof(float), cudaMemcpyDeviceToHost);
    Error(minValue, dim);

    cudaFree(d_values);
    cudaFree(d_minValue);
    free(values);
  }

  return 0;
}
