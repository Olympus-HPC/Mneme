#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)

#ifdef __ENABLE_CUDA__
#define DEV_PREFIX cuda
#elif defined(__ENABLE_HIP__)
#define DEV_PREFIX hip
#endif

#define prefix(name) CAT(DEV_PREFIX, name)

template <typename T>
__global__ void vecAdd_test(T *in, T *out, size_t size, int k) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid] + tid;
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Wrong CLI: Proper one is " << argv[0]
              << " '<num elements>'\n";
  }
  void *deviceAddress = nullptr;

  // Get the address of the device variable
  std::cout << "Device address of deviceVar: " << deviceAddress << std::endl;

  size_t numElements = atoi(argv[1]);
  double *in, *out;
  double val = numElements;
  prefix(Malloc)((void **)&in, numElements * sizeof(double));
  prefix(Malloc)((void **)&out, numElements * sizeof(double));

  for (int i = 0; i < 10; i++) {
    prefix(Memset)(in, 0, numElements * sizeof(double));
    prefix(Memset)(out, 0, numElements * sizeof(double));

    const int threads = 256;
    int num_blocks = (numElements + threads - 1) / threads;
    vecAdd_test<<<num_blocks, threads>>>(in, out, numElements, i % 2);
    prefix(DeviceSynchronize)();
  }

  double *h_in = new double[numElements];
  double *h_out = new double[numElements];
  prefix(Memcpy)(h_in, in, sizeof(double) * numElements, hipMemcpyDeviceToHost);
  prefix(Memcpy)(h_out, out, sizeof(double) * numElements,
                 hipMemcpyDeviceToHost);
  delete[] h_in;
  delete[] h_out;
  prefix(Free)(in);
  prefix(Free)(out);
  return 0;
}
