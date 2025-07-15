#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "StoreTypedefs.h"
#include "StoreElement.h"
#include "kurtosis.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage ./main <elemCount> <repeat>\n";
    return 1;
  }

  const int elemCount = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
    
  storeElement *elem = new storeElement[elemCount];

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(1.f, 2.f);
  for (int i = 0; i < elemCount; i++)
    elem[i] = {i, 0, (unsigned long long int)i, dis(gen)};

  storeElement *d_elem;
  cudaMalloc((void**)&d_elem, sizeof(storeElement) * elemCount);
  cudaMemcpy(d_elem, elem, sizeof(storeElement) * elemCount, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  kurtosisResult* result;
  const size_t s = kurtosis(d_elem, elemCount, repeat, (void**)&result);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total device compute time: " << time * 1e-9 << " (s)\n";
  
  cudaFree(d_elem);

  std::cout << "Results:" << std::endl;
  std::cout << s << " "
            << result->count << " "
            << result->m2 << " "
            << result->m3 << " "
            << result->m4 << std::endl;

  delete [] elem;
  delete result;
  return 0;
}
