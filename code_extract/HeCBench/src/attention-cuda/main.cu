#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
  // input
  float *d_key;
  cudaMalloc((void**)&d_key, n * d * sizeof(float));
  cudaMemcpy(d_key, key, n * d * sizeof(float), cudaMemcpyHostToDevice);

  float *d_value;
  cudaMalloc((void**)&d_value, n * d * sizeof(float));
  cudaMemcpy(d_value, value, n * d * sizeof(float), cudaMemcpyHostToDevice);

  float *d_query;
  cudaMalloc((void**)&d_query, d * sizeof(float));
  cudaMemcpy(d_query, query, d * sizeof(float), cudaMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  cudaMalloc((void**)&d_dot_product, n * sizeof(float));

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  cudaMalloc((void**)&d_output, d * sizeof(float));

  cudaDeviceSynchronize();

  if (impl_num == 3) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      kernel1_warpReduce<<<(n+7)/8, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 2) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      kernel1_warpReduce<<<(n+7)/8, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      kernel2_warpReduce<<<(d+7)/8, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 1) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      kernel1_blockReduce<<<n, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else {
    float *d_score;
    cudaMalloc((void**)&d_score, n * sizeof(float));

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      kernel1<<<(n+255)/256, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      kernel2<<<(n+255)/256, 256>>>(d_exp_sum, d_dot_product, d_score, n);
      kernel3<<<(d+255)/256, 256>>>(d_score, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
    cudaFree(d_score);
  }

  cudaMemcpy(output, d_output, d * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_value);
  cudaFree(d_output);
  cudaFree(d_key);
  cudaFree(d_dot_product);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: fused kernels with block reduce\n");
    printf("implementation 2: fused kernels with warp reduce\n");
    printf("implementation 3: fused kernels with mixed reduce\n");
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (int i = 0; i < n * d; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);

  float* dout = attention_device(key, value, query, n, d, k, r);

  float rmse = 0;
  for (int i = 0; i < d; i++) {
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  }
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
