/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernels.h"

// V: version
template<typename T, int V>
void invokeAddBiasResidualLayerNorm(
          T*     out,
    const T*     input,
    const T*     bias,
    const T*     gamma,
    const T*     beta,
    const float  layernorm_eps,
    int          m,
    int          n)
{
  dim3 grid(m);
  dim3 block(std::min(n, 256));

  if (V == 2) { // assume n % 8 == 0
    addBiasResidualPostLayerNormV2<T><<<grid, n / 8>>>(out, input, bias, gamma, beta, layernorm_eps, n);
  }
  else {
    int num_trips = (n + block.x - 1) / block.x;
    if (num_trips == 1) {
      addBiasResidualPostLayerNorm<T, 1>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else if (num_trips == 2) {
      addBiasResidualPostLayerNorm<T, 2>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else {
      generalAddBiasResidualPostLayerNorm<T>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
  }
}

template<typename T, int V>
void layer(int repeat) {

  const int m = 4096;  // batch size

  int dim[] = {256, 512, 1024, 2048, 4096, 8192};

  for (int i = 0; i < sizeof(dim) / sizeof(int); i++) {

    std::mt19937 gen (19937);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    // n-dimensional data
    const int n = dim[i]; 
    const int input_size = m * n;
    const int output_size = m * n;
    const int input_size_bytes = input_size * sizeof(T);
    const int output_size_bytes = output_size * sizeof(T);
    const int bias_size_bytes = n * sizeof(T);
    const int beta_size_bytes = n * sizeof(T);
    const int gamma_size_bytes = n * sizeof(T);

    T *h_input = (T*) malloc (input_size_bytes);
    T *h_output = (T*) malloc (output_size_bytes);
    T *h_bias = (T*) malloc (bias_size_bytes);
    T *h_gamma = (T*) malloc (gamma_size_bytes);
    T *h_beta = (T*) malloc (beta_size_bytes);

    for (int i = 0; i < input_size; i++) {
      h_input[i] = (T) dis(gen);
    }
    for (int i = 0; i < n; i++) {
      h_bias[i] = (T) dis(gen);
      h_gamma[i] = (T) dis(gen);
      h_beta[i] = (T) dis(gen);
    }

    float layernorm_eps = 1e-6;
    T *d_input, *d_output, *d_bias, *d_gamma, *d_beta;
    cudaMalloc((void**)&d_input, input_size_bytes);
    cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output,  output_size_bytes);
    cudaMemset(d_output,  0, output_size_bytes); // need to reset output

    cudaMalloc((void**)&d_bias,  bias_size_bytes);
    cudaMemcpy(d_bias, h_bias, bias_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_gamma,  gamma_size_bytes);
    cudaMemcpy(d_gamma, h_gamma, gamma_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_beta,  beta_size_bytes);
    cudaMemcpy(d_beta, h_beta, beta_size_bytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      invokeAddBiasResidualLayerNorm<T, V>
          (d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
           m, n, (time * 1e-3f) / repeat);

    cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    float s = 0;
    for (int i = 0; i < output_size; i++)
      s += float(h_output[i]);

    printf("Checksum = %f\n", s / (m * n));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    free(h_input);
    free(h_output);
    free(h_bias);
    free(h_gamma);
    free(h_beta);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  printf("---------------- float16 (version 1) -------------\n");
  layer<half, 1>(repeat);
  printf("---------------- float16 (version 2) -------------\n");
  layer<half, 2>(repeat);

  printf("---------------- bfloat16 (version 1) -------------\n");
  layer<__nv_bfloat16, 1>(repeat);
  printf("---------------- bfloat16 (version 2) -------------\n");
  layer<__nv_bfloat16, 2>(repeat);
  return 0;
}
