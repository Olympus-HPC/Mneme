#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include "kernel.h"

int rounded_division(int number1, int number2) {
  if (number1 % number2 == 0)
    return number1 / number2;
  return number1 / number2 + 1;
}

template <typename T>
void contract (const int max_N, const int max_C, const int repeat) {
  // tensor
  const size_t tensor_size = (size_t)max_N * max_N * max_N * max_C;
  const size_t tensor_size_byte = tensor_size * sizeof(T);

  T* tensor_value = (T*) malloc (tensor_size_byte);
  for (size_t i = 0; i < tensor_size; i++)
    tensor_value[i] = 1;

  T* device_tensor_value;
  cudaMalloc(&device_tensor_value, tensor_size_byte);

  // adjacency matrix
  const size_t adj_size = max_N * max_N;
  const size_t adj_size_byte = adj_size * sizeof(T);
  
  // longest kernel time occurs when all values in adj_value are positive
  T* adj_value = (T*) malloc (adj_size_byte);
  for (size_t i = 0; i < adj_size; i++) adj_value[i] = 1;

  T* device_adj_value;
  cudaMalloc((void**)&device_adj_value, adj_size_byte);

  // output value 
  const size_t output_size = max_N * max_N * max_C * nContractions;
  const size_t output_size_byte = max_N * max_N * max_C * nContractions * sizeof(T);

  T* value = (T*) malloc (output_size_byte);

  T* device_value;
  cudaMalloc((void**)&device_value, output_size_byte);

  // launch kernel
  cudaMemcpy(device_tensor_value, tensor_value, tensor_size_byte, cudaMemcpyHostToDevice);
  cudaMemcpy(device_adj_value, adj_value, adj_size_byte, cudaMemcpyHostToDevice);

  const int nThreads = 256;
  dim3 dimGrid(rounded_division(output_size, nThreads));
  dim3 dimBlock(nThreads);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    contraction <<< dimGrid, dimBlock >>> (
      device_tensor_value, device_adj_value, device_value, output_size, max_N, max_C);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(value, device_value, output_size_byte, cudaMemcpyDeviceToHost);

  double checksum = 0;
  for (size_t i = 0; i < output_size; i++) checksum += value[i];
  printf("Checksum: %lf min:%lf max:%lf\n", checksum, 
         *std::min_element(value, value+output_size),
         *std::max_element(value, value+output_size));

  cudaFree(device_value);
  cudaFree(device_tensor_value);
  cudaFree(device_adj_value);
  free(value);
  free(tensor_value);
  free(adj_value);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }
 
  int max_N = atoi(argv[1]);
  int max_C = nContractions;
  int repeat = atoi(argv[2]);

  contract<float>(max_N, max_C, repeat);
  contract<double>(max_N, max_C, repeat);

  return 0;
}
