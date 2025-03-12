#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

__device__ int some_value = 1.0;

template <typename T> __global__ 
void vecAdd_test(T *in, T *out, size_t size, int k) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid] + tid; // + some_value;
  }
}

int main(int argc, const char *argv[]) {

  void* deviceAddress = nullptr;

  // Get the address of the device variable
  hipError_t err = hipGetSymbolAddress(&deviceAddress, HIP_SYMBOL(some_value));
  std::cout << "Device address of deviceVar: " << deviceAddress << std::endl;


  size_t numElements = atoi(argv[1]);
  double *in, *out;
  double val = numElements;
  hipMalloc((void **)&in, numElements * sizeof(double));
  hipMalloc((void **)&out, numElements * sizeof(double));

  for (int i = 0; i < 10 ; i++){
    hipMemset(in, 0, numElements * sizeof(double));
    hipMemset(out, 0, numElements * sizeof(double));

    const int threads = 256;
    int num_blocks = (numElements + threads - 1) / threads;
    vecAdd_test<<< num_blocks, threads>>>(in, out, numElements, i % 2);
    hipDeviceSynchronize();
  }

  double *h_in = new double[numElements];
  double *h_out = new double[numElements];
  hipMemcpy(h_in, in, sizeof(double)*numElements, hipMemcpyDeviceToHost);
  hipMemcpy(h_out, out, sizeof(double)*numElements, hipMemcpyDeviceToHost);
  int ret = 0;
  for (int i = 0; i < numElements; i++){
    if (h_in[i] + i != h_out[i]){
      std::cout << "Values at " << i << " differ\n";
      std::cout << "Values " << h_in[i] << " " << h_out[i] << "differ\n";
      ret = -1;
      break;
    }
  }

  delete [] h_in;
  delete [] h_out;
  hipFree(in);
  hipFree(out);
  return ret;
}
